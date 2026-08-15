"""Boucle d'entraînement, en PyTorch pur.

Aucun framework d'entraînement : ni HuggingFace Trainer, ni Lightning. Chaque
ligne est lisible, ce qui est précisément l'intérêt du projet, on veut pouvoir
suivre ce qui se passe plutôt que de faire confiance à une abstraction.

Trois modes :

    # Test de surapprentissage : 100 parties, la loss doit tomber vers zéro.
    # C'est le garde-fou obligatoire avant le run long.
    python train.py --overfit

    # Entraînement complet
    python train.py --run-name run1

    # Reprise après interruption
    python train.py --run-name run1 --resume
"""

import argparse
import csv
import json
import math
import os
import time
from contextlib import nullcontext

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import numpy as np
import torch

from model import ChessGPT, ModelConfig

# TFLOPS bf16 réellement mesurés sur la carte au 2026-08-04 (plafond 280 W).
# On calcule le MFU par rapport à ce chiffre et non par rapport à la fiche
# technique du constructeur : c'est la seule référence honnête, puisque c'est
# la performance dont on dispose réellement.
MEASURED_PEAK_TFLOPS_BF16 = 62.86


# ---------------------------------------------------------------------------
# Données
# ---------------------------------------------------------------------------

class MemmapDataset:
    """Échantillonne des fenêtres de `block_size` tokens dans un .bin uint16.

    Le corpus est un unique tableau plat où toutes les parties sont
    concaténées, séparées par <eos><bos>. Pour former un batch, on tire des
    positions au hasard et on lit les tokens consécutifs à partir de là.

    Une fenêtre peut donc chevaucher deux parties. C'est voulu : avec une
    moyenne de 73.5 tokens par partie et un contexte de 256, découper
    proprement une partie par séquence gaspillerait 70 % du calcul en
    remplissage. Le modèle apprend simplement que <eos><bos> signifie « tout
    ce qui précède ne compte plus », ce qu'il doit de toute façon apprendre
    pour savoir reconnaître une fin de partie.
    """

    def __init__(self, path: str, block_size: int, device: str):
        self.path = path
        self.block_size = block_size
        self.device = device
        # On rouvre le memmap à chaque époque plutôt que de garder une
        # référence : c'est la recommandation de numpy pour éviter une fuite
        # de descripteurs sur les fichiers volumineux.
        self.n_tokens = os.path.getsize(path) // 2

    def get_batch(self, batch_size: int, generator=None):
        data = np.memmap(self.path, dtype=np.uint16, mode="r")
        # -1 car il faut un token de plus pour la cible décalée
        high = len(data) - self.block_size - 1
        ix = torch.randint(high, (batch_size,), generator=generator)
        x = torch.stack([
            torch.from_numpy(data[i:i + self.block_size].astype(np.int64))
            for i in ix])
        y = torch.stack([
            torch.from_numpy(data[i + 1:i + 1 + self.block_size].astype(np.int64))
            for i in ix])
        # pin_memory + non_blocking : le transfert vers le GPU se recouvre avec
        # le calcul du batch précédent au lieu de le bloquer.
        return (x.pin_memory().to(self.device, non_blocking=True),
                y.pin_memory().to(self.device, non_blocking=True))


class OverfitDataset:
    """Un jeu minuscule gardé entièrement en mémoire GPU.

    Sert au test de surapprentissage : si le modèle n'arrive pas à apprendre
    par cœur cent parties, c'est qu'il y a un bug quelque part, dans le
    modèle, dans les données, ou dans la boucle. Autant le découvrir en trois
    minutes plutôt qu'après une nuit de calcul.
    """

    def __init__(self, games_path: str, vocab_path: str, block_size: int,
                 n_games: int, device: str):
        with open(vocab_path) as f:
            vocab = json.load(f)
        stoi, bos, eos = vocab["stoi"], vocab["bos_id"], vocab["eos_id"]

        ids = []
        with open(games_path) as f:
            for i, line in enumerate(f):
                if i >= n_games:
                    break
                ids.append(bos)
                ids.extend(stoi[m] for m in line.split())
                ids.append(eos)

        self.data = torch.tensor(ids, dtype=torch.int64, device=device)
        self.block_size = block_size
        self.device = device
        self.n_tokens = len(ids)
        self.n_games = min(n_games, i + 1)

    def get_batch(self, batch_size: int, generator=None):
        high = len(self.data) - self.block_size - 1
        ix = torch.randint(high, (batch_size,), device=self.device)
        x = torch.stack([self.data[i:i + self.block_size] for i in ix])
        y = torch.stack([self.data[i + 1:i + 1 + self.block_size] for i in ix])
        return x, y


# ---------------------------------------------------------------------------
# Planification du taux d'apprentissage
# ---------------------------------------------------------------------------

def get_lr(step: int, cfg) -> float:
    """Chauffe linéaire puis décroissance en cosinus.

    Pourquoi une chauffe : au tout début, les gradients sont énormes et
    désordonnés. Appliquer immédiatement le taux d'apprentissage nominal fait
    diverger le modèle, ou pire, le fait converger vers une solution médiocre
    dont il ne ressortira pas. On monte donc progressivement.

    Pourquoi un cosinus ensuite : en fin d'entraînement on veut de tout petits
    pas, pour se poser dans un minimum plutôt que de rebondir autour. Le
    cosinus donne une décroissance douce au début et de plus en plus marquée,
    ce qui empiriquement bat la décroissance linéaire.
    """
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    if step >= cfg.max_steps:
        return cfg.lr_min
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.lr_min + coeff * (cfg.lr - cfg.lr_min)


# ---------------------------------------------------------------------------
# Optimiseur
# ---------------------------------------------------------------------------

def build_optimizer(model, lr, weight_decay, betas, device_type):
    """AdamW, avec weight decay uniquement sur les matrices.

    Le weight decay tire les poids vers zéro pour limiter le surapprentissage.
    On l'applique aux matrices de projection, mais pas aux vecteurs de gain des
    normalisations ni aux biais : ces paramètres-là contrôlent une échelle, et
    les pousser vers zéro revient à éteindre progressivement des couches
    entières. C'est une distinction que beaucoup d'implémentations oublient.
    """
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 else no_decay).append(p)

    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    # fused=True fait tourner la mise à jour d'Adam dans un seul noyau CUDA
    # au lieu d'un par tenseur : sur un modèle à 300 tenseurs, ça compte.
    use_fused = device_type == "cuda"
    opt = torch.optim.AdamW(groups, lr=lr, betas=betas, fused=use_fused)
    return opt, len(decay), len(no_decay)


# ---------------------------------------------------------------------------
# Évaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def estimate_loss(model, dataset, batch_size, n_batches, ctx):
    model.eval()
    losses = torch.zeros(n_batches)
    for i in range(n_batches):
        x, y = dataset.get_batch(batch_size)
        with ctx:
            _, loss = model(x, y)
        losses[i] = loss.item()
    model.train()
    return losses.mean().item()


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------

def save_snapshot(path, raw_model, step, tokens_seen, best_val):
    """Instantané léger, pour l'évaluation des checkpoints intermédiaires.

    Différence avec un checkpoint de reprise : on ne sauvegarde **que** les
    poids, sans l'état de l'optimiseur. Celui-ci pèse deux fois le modèle (les
    deux moments d'Adam) et ne sert qu'à reprendre un entraînement, jamais à
    évaluer. Un instantané fait donc 210 Mo au lieu de 630.

    Ces instantanés sont numérotés et ne s'écrasent pas, contrairement au
    checkpoint de reprise. C'est ce qui permet de tracer a posteriori la
    courbe « taux de coups légaux en fonction des tokens vus », le graphique
    le plus intéressant du projet, et qu'on ne peut pas reconstituer si on a
    écrasé les états intermédiaires.
    """
    tmp = path + ".tmp"
    torch.save({
        "model": raw_model.state_dict(),
        "step": step,
        "tokens_seen": tokens_seen,
        "model_config": raw_model.cfg.__dict__,
        "best_val": best_val,
        "snapshot": True,
    }, tmp)
    os.replace(tmp, path)


def save_checkpoint(path, raw_model, optimizer, step, tokens_seen, cfg,
                    best_val, elapsed):
    """Sauvegarde tout ce qu'il faut pour reprendre exactement où on en est.

    Sauvegarder les poids ne suffit pas : sans l'état de l'optimiseur (les deux
    moments d'Adam, qui sont une moyenne mobile des gradients passés), une
    reprise repart avec un optimiseur amnésique et la loss fait un bond visible
    pendant quelques centaines de pas.

    On écrit d'abord dans un fichier temporaire, puis on renomme. Le renommage
    est atomique sur un système de fichiers POSIX : si le processus est tué
    pendant l'écriture, on garde le checkpoint précédent intact plutôt que de
    se retrouver avec un fichier tronqué.
    """
    tmp = path + ".tmp"
    torch.save({
        "model": raw_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "tokens_seen": tokens_seen,
        "model_config": raw_model.cfg.__dict__,
        "train_config": vars(cfg),
        "best_val": best_val,
        "elapsed": elapsed,
        "torch_version": torch.__version__,
    }, tmp)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-name", default="run1")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out-dir", default="checkpoints")
    p.add_argument("--log-dir", default="logs")
    p.add_argument("--device", default="cuda:1",
                   help="cuda:1 est le RTX 3090 avec CUDA_DEVICE_ORDER=PCI_BUS_ID")

    # Modèle
    p.add_argument("--n-layer", type=int, default=16)
    p.add_argument("--n-head", type=int, default=8)
    p.add_argument("--n-embd", type=int, default=512)
    p.add_argument("--mlp-hidden", type=int, default=1408)
    p.add_argument("--block-size", type=int, default=256)

    # Optimisation
    p.add_argument("--batch-size", type=int, default=192,
                   help="séquences par micro-batch")
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=6e-4)
    p.add_argument("--lr-min", type=float, default=6e-5)
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--max-steps", type=int, default=16000)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.95)
    p.add_argument("--grad-clip", type=float, default=1.0)

    # Journalisation et checkpoints
    p.add_argument("--eval-interval", type=int, default=250)
    p.add_argument("--eval-batches", type=int, default=50)
    p.add_argument("--log-interval", type=int, default=10)
    p.add_argument("--ckpt-minutes", type=float, default=30.0)
    p.add_argument("--snapshot-every", type=int, default=1000,
                   help="instantané numéroté tous les N steps (0 pour désactiver)")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--compile", dest="compile", action="store_true", default=True)
    p.add_argument("--no-compile", dest="compile", action="store_false")

    # Test de surapprentissage
    p.add_argument("--overfit", action="store_true")
    p.add_argument("--overfit-games", type=int, default=100)
    p.add_argument("--overfit-steps", type=int, default=1200)
    p.add_argument("--overfit-target", type=float, default=0.05,
                   help="loss en dessous de laquelle le test est considéré réussi")

    cfg = p.parse_args()

    torch.manual_seed(1337)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device = cfg.device
    device_type = "cuda" if device.startswith("cuda") else "cpu"
    # bf16 : même plage d'exposants que le float32, donc pas de risque de
    # débordement, et pas besoin du GradScaler qu'imposait le fp16.
    ctx = (torch.autocast(device_type=device_type, dtype=torch.bfloat16)
           if device_type == "cuda" else nullcontext())

    # Le vocabulaire est cherché d'abord dans le répertoire de données, puis
    # dans `data/`. Sans ce repli, entraîner sur un corpus rangé dans un
    # sous-répertoire (data/4mois/) échoue au démarrage parce que vocab.json
    # est resté à la racine : ce qui est exactement arrivé, après trois heures
    # de préparation de données réussies.
    for candidat in (os.path.join(cfg.data_dir, "vocab.json"),
                     os.path.join("data", "vocab.json")):
        if os.path.exists(candidat):
            with open(candidat) as f:
                vocab_size = json.load(f)["vocab_size"]
            print(f"[vocab] {candidat} ({vocab_size} tokens)")
            break
    else:
        raise SystemExit(
            f"vocab.json introuvable, ni dans {cfg.data_dir} ni dans data/")

    model_cfg = ModelConfig(
        vocab_size=vocab_size,
        block_size=cfg.block_size,
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
        n_embd=cfg.n_embd,
        mlp_hidden=cfg.mlp_hidden,
    )

    if cfg.overfit:
        cfg.max_steps = cfg.overfit_steps
        cfg.warmup_steps = min(cfg.warmup_steps, cfg.overfit_steps // 10)
        cfg.eval_interval = max(10, cfg.overfit_steps // 40)

    # --- Données ---
    if cfg.overfit:
        train_ds = OverfitDataset(
            os.path.join(cfg.data_dir, "games_uci.txt"),
            os.path.join(cfg.data_dir, "vocab.json"),
            cfg.block_size, cfg.overfit_games, device)
        val_ds = train_ds
        print(f"[données] mode surapprentissage : {train_ds.n_games} parties, "
              f"{train_ds.n_tokens:,} tokens, gardés en mémoire GPU")
    else:
        train_ds = MemmapDataset(os.path.join(cfg.data_dir, "train.bin"),
                                 cfg.block_size, device)
        val_ds = MemmapDataset(os.path.join(cfg.data_dir, "val.bin"),
                               cfg.block_size, device)
        print(f"[données] train {train_ds.n_tokens:,} tokens | "
              f"val {val_ds.n_tokens:,} tokens")

    # --- Modèle ---
    model = ChessGPT(model_cfg).to(device)
    raw_model = model          # référence non compilée, pour les checkpoints
    n_params = model.num_params()
    flops_per_token = model.flops_per_token()
    print(f"[modèle] {n_params:,} paramètres non-embedding | "
          f"{flops_per_token:,.0f} FLOP/token")

    optimizer, n_decay, n_nodecay = build_optimizer(
        model, cfg.lr, cfg.weight_decay, (cfg.beta1, cfg.beta2), device_type)
    print(f"[optim] AdamW fused | {n_decay} tenseurs avec weight decay, "
          f"{n_nodecay} sans")

    # --- Reprise ---
    step = 0
    tokens_seen = 0
    best_val = float("inf")
    elapsed_before = 0.0
    ckpt_path = os.path.join(cfg.out_dir, f"{cfg.run_name}_last.pt")
    best_path = os.path.join(cfg.out_dir, f"{cfg.run_name}_best.pt")

    if cfg.resume:
        if not os.path.exists(ckpt_path):
            raise SystemExit(f"--resume demandé mais {ckpt_path} est absent")
        print(f"[reprise] chargement de {ckpt_path}")
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        raw_model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        step = ck["step"]
        tokens_seen = ck["tokens_seen"]
        best_val = ck["best_val"]
        elapsed_before = ck["elapsed"]
        print(f"[reprise] step {step}, {tokens_seen:,} tokens déjà vus, "
              f"{elapsed_before/3600:.2f} h de calcul cumulées")

    if cfg.compile:
        print("[compile] torch.compile en cours (première itération lente)...",
              flush=True)
        model = torch.compile(model)

    os.makedirs(cfg.out_dir, exist_ok=True)
    os.makedirs(cfg.log_dir, exist_ok=True)

    # --- Journal CSV ---
    csv_path = os.path.join(cfg.log_dir, f"{cfg.run_name}_metrics.csv")
    new_csv = not (cfg.resume and os.path.exists(csv_path))
    csv_file = open(csv_path, "a", newline="")
    writer = csv.writer(csv_file)
    if new_csv:
        writer.writerow(["step", "tokens_vus", "loss_train", "loss_val",
                         "lr", "tokens_par_s", "mfu", "secondes_ecoulees",
                         "grad_norm"])
        csv_file.flush()

    tokens_per_step = cfg.batch_size * cfg.block_size * cfg.grad_accum
    print(f"[plan] {tokens_per_step:,} tokens/step x {cfg.max_steps:,} steps "
          f"= {tokens_per_step * cfg.max_steps / 1e6:.1f} M tokens")

    # --- Boucle ---
    t_start = time.perf_counter()
    t_last_ckpt = t_start
    t_log = time.perf_counter()
    tokens_since_log = 0
    loss_train_running = float("nan")
    overfit_reached = False

    model.train()
    while step < cfg.max_steps:
        lr = get_lr(step, cfg)
        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        loss_accum = 0.0
        for micro in range(cfg.grad_accum):
            x, y = train_ds.get_batch(cfg.batch_size)
            with ctx:
                _, loss = model(x, y)
                loss = loss / cfg.grad_accum
            loss.backward()
            loss_accum += loss.item()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                   cfg.grad_clip)
        optimizer.step()

        step += 1
        tokens_seen += tokens_per_step
        tokens_since_log += tokens_per_step
        loss_train_running = loss_accum

        # --- Journalisation périodique ---
        if step % cfg.log_interval == 0:
            torch.cuda.synchronize(device) if device_type == "cuda" else None
            now = time.perf_counter()
            dt = now - t_log
            tok_per_s = tokens_since_log / dt
            # MFU : la part de la puissance réellement mesurée du GPU qu'on
            # exploite. Numérateur = FLOP effectivement utiles par seconde.
            mfu = (tok_per_s * flops_per_token) / (MEASURED_PEAK_TFLOPS_BF16 * 1e12)
            elapsed = elapsed_before + (now - t_start)
            print(f"step {step:>6} | loss {loss_accum:6.4f} | lr {lr:.2e} | "
                  f"{tok_per_s:>9,.0f} tok/s | MFU {mfu:5.1%} | "
                  f"{tokens_seen/1e6:7.1f} M tokens | {elapsed/60:6.1f} min",
                  flush=True)
            writer.writerow([step, tokens_seen, f"{loss_accum:.6f}", "",
                             f"{lr:.6e}", f"{tok_per_s:.1f}", f"{mfu:.4f}",
                             f"{elapsed:.1f}", f"{grad_norm:.4f}"])
            csv_file.flush()
            t_log = now
            tokens_since_log = 0

        # --- Évaluation ---
        if step % cfg.eval_interval == 0 or step == cfg.max_steps:
            val_loss = estimate_loss(model, val_ds, cfg.batch_size,
                                     cfg.eval_batches, ctx)
            elapsed = elapsed_before + (time.perf_counter() - t_start)
            print(f"  >> eval step {step}: loss_val {val_loss:.4f} "
                  f"(train {loss_train_running:.4f})", flush=True)
            writer.writerow([step, tokens_seen, f"{loss_train_running:.6f}",
                             f"{val_loss:.6f}", f"{lr:.6e}", "", "",
                             f"{elapsed:.1f}", ""])
            csv_file.flush()

            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint(best_path, raw_model, optimizer, step,
                                tokens_seen, cfg, best_val, elapsed)

            if cfg.overfit and loss_train_running < cfg.overfit_target:
                overfit_reached = True
                print(f"\n[surapprentissage] loss {loss_train_running:.4f} "
                      f"< {cfg.overfit_target} au step {step}, le test passe.")
                break

            # On remet à zéro les compteurs de débit après une évaluation.
            # Sinon l'intervalle de mesure suivant englobe le temps passé à
            # évaluer, et le débit rapporté s'effondre artificiellement, ce
            # qui donnait des lignes à 15 % de MFU alternant avec des lignes à
            # 55 %, sans que rien n'ait changé dans l'entraînement lui-même.
            t_log = time.perf_counter()
            tokens_since_log = 0

        # --- Instantané numéroté, pour la courbe de progression ---
        if (not cfg.overfit and cfg.snapshot_every
                and step % cfg.snapshot_every == 0):
            snap = os.path.join(cfg.out_dir,
                                f"{cfg.run_name}_step{step:06d}.pt")
            save_snapshot(snap, raw_model, step, tokens_seen, best_val)
            print(f"  [instantané] {snap} "
                  f"({os.path.getsize(snap)/1e6:.0f} Mo)", flush=True)

        # --- Checkpoint périodique, à l'horloge ---
        now = time.perf_counter()
        if not cfg.overfit and (now - t_last_ckpt) >= cfg.ckpt_minutes * 60:
            elapsed = elapsed_before + (now - t_start)
            save_checkpoint(ckpt_path, raw_model, optimizer, step, tokens_seen,
                            cfg, best_val, elapsed)
            t_last_ckpt = now
            print(f"  [checkpoint] step {step} -> {ckpt_path} "
                  f"({os.path.getsize(ckpt_path)/1e6:.0f} Mo)", flush=True)

    # --- Fin ---
    elapsed = elapsed_before + (time.perf_counter() - t_start)
    if not cfg.overfit:
        save_checkpoint(ckpt_path, raw_model, optimizer, step, tokens_seen,
                        cfg, best_val, elapsed)
    csv_file.close()

    summary = {
        "run_name": cfg.run_name,
        "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mode": "surapprentissage" if cfg.overfit else "entrainement",
        "steps": step,
        "tokens_vus": tokens_seen,
        "loss_train_finale": loss_train_running,
        "best_val_loss": None if best_val == float("inf") else best_val,
        "duree_s": round(elapsed, 1),
        "parametres_non_embedding": n_params,
        "flops_par_token": flops_per_token,
        "peak_tflops_reference": MEASURED_PEAK_TFLOPS_BF16,
        "config": vars(cfg),
    }
    if cfg.overfit:
        summary["test_reussi"] = overfit_reached
        summary["cible_loss"] = cfg.overfit_target
        summary["parties"] = train_ds.n_games
        summary["tokens_du_jeu"] = train_ds.n_tokens

    suffix = "overfit" if cfg.overfit else "train"
    out = os.path.join(cfg.log_dir, f"{cfg.run_name}_{suffix}_summary.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n=== Terminé : {step} steps, {tokens_seen/1e6:.1f} M tokens, "
          f"{elapsed/60:.1f} min ===")
    print(f"résumé -> {out}")

    if cfg.overfit and not overfit_reached:
        raise SystemExit(
            f"\nECHEC : la loss n'est pas descendue sous {cfg.overfit_target} "
            f"en {cfg.overfit_steps} steps (elle vaut {loss_train_running:.4f}). "
            f"Il y a un bug, ne pas lancer le run complet.")


if __name__ == "__main__":
    main()
