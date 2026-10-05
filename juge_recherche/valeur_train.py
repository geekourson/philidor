"""Réseau de valeur, étape 2 : une copie de run2 affinée entièrement pour noter.

La politique n'est pas touchée : c'est une copie séparée de run2_best.pt, avec
une tête scalaire sur la sortie de `norm_final`. Au token du coup i, elle
prédit sigmoid(cp · ln10 / 400), la note Stockfish de la position obtenue,
vue par le joueur qui vient de jouer (voir valeur_encoder.py).

BOURRAGE À DROITE, ET POURQUOI IL EST SANS DANGER ICI. Le modèle n'a pas de
masque de bourrage ; le bourrage à GAUCHE du diagnostic décalait tout. À droite,
l'attention causale fait qu'aucun token réel ne voit les <pad> qui le suivent,
et les positions RoPE restent celles de la partie seule. Contrôle intégré au
démarrage : notes calculées en lot bourré contre une séquence à la fois.

Le verrou est intégré : à chaque évaluation, taux de gaffes hors ligne sur les
positions du bot (réglage d'alpha) et sur les 5 000 du diagnostic (publication).
Repères diagnostic : argmax 18,32 % | juge 15,72 % | Stockfish p1 14,94 % |
p4 11,88 %.
"""
import argparse, csv, json, math, os, time
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import ChessGPT, ModelConfig

SANS = -32768
K = math.log(10) / 400


def paris():
    return datetime.now(ZoneInfo("Europe/Paris")).strftime("%H:%M")


class Valeur(nn.Module):
    def __init__(self, gpt):
        super().__init__()
        self.gpt = gpt
        self.tete = nn.Linear(gpt.cfg.n_embd, 1)
        nn.init.zeros_(self.tete.weight); nn.init.zeros_(self.tete.bias)

    def forward(self, idx):
        g = self.gpt; T = idx.shape[1]
        x = g.tok_emb(idx); cos, sin = g.rope_cos[:T], g.rope_sin[:T]
        for b in g.blocks:
            x = b(x, cos, sin)
        return self.tete(g.norm_final(x)).squeeze(-1)       # logit, (B, T)


def lots(longueurs, idx, budget, rng):
    """Lots par longueurs voisines, B x Lmax <= budget ; ordre des lots mélangé."""
    idx = idx[np.argsort(longueurs[idx] + rng.random(len(idx)), kind="stable")]
    out, cur, lmax = [], [], 0
    for i in idx:
        L = int(longueurs[i])
        if cur and max(lmax, L) * (len(cur) + 1) > budget:
            out.append(cur); cur, lmax = [], 0
        cur.append(int(i)); lmax = max(lmax, L)
    if cur:
        out.append(cur)
    rng.shuffle(out)
    return out


def tenseurs(ids, T, C, O, pad, dev):
    L = max(int(O[i + 1] - O[i]) for i in ids)
    x = np.full((len(ids), L), pad, np.int64); y = np.full((len(ids), L), SANS, np.int32)
    for r, i in enumerate(ids):
        a, b = O[i], O[i + 1]
        x[r, :b - a] = T[a:b]; y[r, :b - a] = C[a:b]
    x = torch.from_numpy(x).to(dev, non_blocking=True)
    y = torch.from_numpy(y).to(dev, non_blocking=True)
    m = y != SANS
    cible = torch.sigmoid(y.float().clamp(-10000, 10000) * K)
    return x, cible, m


def perte(logit, cible, m):
    l = F.binary_cross_entropy_with_logits(logit.float(), cible, reduction="none")
    return (l * m).sum() / m.sum().clamp(min=1)


# ---------------------------------------------------------------- le verrou

def charger_verrou(positions, modele, sf, stoi, bos, ctx):
    P = {json.loads(l)["fen"]: json.loads(l)["prefixe"] for l in open(positions)}
    S = {json.loads(l)["fen"]: json.loads(l)["cp"] for l in open(sf)}
    racines = []
    for l in open(modele):
        m = json.loads(l)
        if m["fen"] not in P or m["fen"] not in S:
            continue
        base = [bos] + [stoi[u] for u in P[m["fen"]]]
        cands = m["classement"][:5]
        racines.append({"cp": S[m["fen"]], "cands": cands,
                        "lp": [math.log(max(x, 1e-9)) for x in m["p"][:5]],
                        "seqs": [(base + [stoi[u]])[-ctx:] for u in cands]})
    return racines


@torch.no_grad()
def noter(modele, racines, pad, dev, lot=256):
    seqs = [s for r in racines for s in r["seqs"]]
    out = np.zeros(len(seqs), np.float32)
    ordre = np.argsort([len(s) for s in seqs])
    for k in range(0, len(seqs), lot):
        b = ordre[k:k + lot]; L = max(len(seqs[i]) for i in b)
        x = torch.full((len(b), L), pad, dtype=torch.long)
        for r, i in enumerate(b):
            x[r, :len(seqs[i])] = torch.tensor(seqs[i])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg = modele(x.to(dev)).float()
        fin = torch.tensor([len(seqs[i]) - 1 for i in b], device=dev)
        out[b] = lg[torch.arange(len(b), device=dev), fin].cpu().numpy()
    j = 0
    for r in racines:
        r["v"] = out[j:j + len(r["seqs"])].tolist(); j += len(r["seqs"])


def taux(racines, k, alpha):
    g = n = 0
    for r in racines:
        c = [(u, v, lp) for u, v, lp in zip(r["cands"][:k], r["v"][:k], r["lp"][:k]) if u in r["cp"]]
        if not c:
            continue
        choix = max(c, key=lambda t: t[1] + alpha * t[2])[0]
        n += 1; g += (max(r["cp"].values()) - r["cp"][choix]) >= 100
    return g, n


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0, c - h), min(1, c + h)


def verrou(modele, diag, prop, pad, dev):
    modele.eval(); noter(modele, diag, pad, dev); noter(modele, prop, pad, dev); modele.train()
    ALPHAS = (0.0, 0.1, 0.25, 0.5, 1.0, 2.0)
    reg = {al: taux(prop, 5, al) for al in ALPHAS}
    best = min(ALPHAS, key=lambda al: reg[al][0] / reg[al][1])
    g0, n0 = taux(diag, 5, 0.0); g, n = taux(diag, 5, best)
    return {"seul_top5": g0 / n0, "alpha": best, "diag": wilson(g, n), "n": n,
            "top2": (lambda t: t[0] / t[1])(taux(diag, 2, 0.0)),
            "propres_alpha": reg[best][0] / reg[best][1]}


# ---------------------------------------------------------------- entraînement

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/valeur")
    p.add_argument("--ckpt", default="checkpoints/run2_best.pt")
    p.add_argument("--vocab", default="data/vocab.json")
    p.add_argument("--out", default="checkpoints/valeur.pt")
    p.add_argument("--log", default="logs/valeur/train.csv")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--budget", type=int, default=16384, help="tokens (bourrage compris) par micro-lot")
    p.add_argument("--accum", type=int, default=2)
    p.add_argument("--epoques", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lr-tete", type=float, default=1e-3)
    p.add_argument("--lr-min-frac", type=float, default=0.1)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--wd", type=float, default=0.1)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--max-steps", type=int, default=0, help="arrêt anticipé (essai)")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--reprendre", default="")
    p.add_argument("--init", default="", help="poids de depart (valeur.pt), optimiseur neuf")
    a = p.parse_args()
    dev = a.device
    torch.manual_seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = True

    v = json.load(open(a.vocab)); stoi, bos, pad = v["stoi"], v["bos_id"], v["pad_id"]
    T = np.fromfile(os.path.join(a.data, "tokens.u16"), np.uint16)
    C = np.fromfile(os.path.join(a.data, "cp.i16"), np.int16)
    O = np.fromfile(os.path.join(a.data, "offsets.i64"), np.int64)
    S = np.fromfile(os.path.join(a.data, "split.u8"), np.uint8)
    Lg = np.diff(O)
    itr, iva = np.where(S == 0)[0], np.where(S == 1)[0]
    print(f"[{paris()}] {len(itr):,} parties d'entrainement, {len(iva):,} de validation, "
          f"{int((S==2).sum()):,} exclues", flush=True)

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = ModelConfig(**ck["model_config"])
    assert cfg.vocab_size == len(v["itos"]) == ck["model"]["tok_emb.weight"].shape[0]
    gpt = ChessGPT(cfg); gpt.load_state_dict(ck["model"])
    modele = Valeur(gpt).to(dev)

    diag = charger_verrou("diag/positions.jsonl", "diag/modele.jsonl", "diag/stockfish18.jsonl",
                          stoi, bos, cfg.block_size)
    prop = charger_verrou("diag/propres_positions.jsonl", "diag/propres_modele.jsonl",
                          "diag/propres_sf18.jsonl", stoi, bos, cfg.block_size)
    print(f"[{paris()}] verrou : {len(diag)} positions diagnostic, {len(prop)} positions du bot", flush=True)

    # contrôle du bourrage à droite : lot bourré contre une séquence à la fois
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        modele.tete.weight.normal_(0, 0.02)
        ids = [int(i) for i in iva[:16]]
        x, _, m = tenseurs(ids, T, C, O, pad, dev)
        lot = modele(x).float()
        ecart = 0.0
        for r, i in enumerate(ids):
            seul = modele(torch.from_numpy(T[O[i]:O[i + 1]].astype(np.int64))[None].to(dev)).float()[0]
            ecart = max(ecart, float((seul - lot[r, :len(seul)]).abs().max()))
        nn.init.zeros_(modele.tete.weight)
    print(f"[controle] bourrage a droite : ecart max lot / une par une = {ecart:.4f} (logit)", flush=True)
    assert ecart < 0.05, "le bourrage a droite change les notes : arret"

    dec, nodec = [], []
    for n_, p_ in modele.gpt.named_parameters():
        (dec if p_.dim() >= 2 else nodec).append(p_)
    opt = torch.optim.AdamW([{"params": dec, "weight_decay": a.wd, "lr": a.lr},
                             {"params": nodec, "weight_decay": 0.0, "lr": a.lr},
                             {"params": modele.tete.parameters(), "weight_decay": 0.0, "lr": a.lr_tete}],
                            betas=(0.9, 0.95), fused=True)
    base_lr = [g["lr"] for g in opt.param_groups]

    rng = np.random.default_rng(a.seed)
    micro = lots(Lg, itr, a.budget, rng)
    n_ep = a.epoques
    total = int(len(micro) * n_ep) // a.accum
    if a.max_steps:
        total = min(total, a.max_steps)
    val_lots = lots(Lg, iva, a.budget, np.random.default_rng(0))
    print(f"[{paris()}] {len(micro):,} micro-lots par epoque -> {total:,} pas "
          f"(accum {a.accum}, budget {a.budget})", flush=True)

    step, best = 0, float("inf")
    if a.init:
        e0 = torch.load(a.init, map_location="cpu", weights_only=False)
        modele.load_state_dict(e0["modele"])
        print(f"[{paris()}] poids de depart : {a.init} (pas {e0['step']})", flush=True)
    if a.reprendre:
        e = torch.load(a.reprendre, map_location="cpu", weights_only=False)
        modele.load_state_dict(e["modele"]); opt.load_state_dict(e["opt"])
        step, best = e["step"], e["best"]
        print(f"[{paris()}] reprise au pas {step}", flush=True)

    def lr_a(s):
        if s < a.warmup:
            return (s + 1) / a.warmup
        f = (s - a.warmup) / max(1, total - a.warmup)
        return a.lr_min_frac + (1 - a.lr_min_frac) * 0.5 * (1 + math.cos(math.pi * min(1.0, f)))

    @torch.no_grad()
    def val():
        modele.eval(); tot = nb = 0.0
        for b in val_lots:
            x, c, m = tenseurs(b, T, C, O, pad, dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg = modele(x).float()
            l = F.binary_cross_entropy_with_logits(lg, c, reduction="none")
            tot += float((l * m).sum()); nb += float(m.sum())
        modele.train()
        return tot / nb

    os.makedirs(os.path.dirname(a.log), exist_ok=True)
    nouveau = not (a.reprendre and os.path.exists(a.log))
    fcsv = open(a.log, "a" if not nouveau else "w", newline=""); w = csv.writer(fcsv)
    if nouveau:
        w.writerow(["step", "lr", "perte_train", "perte_val", "diag_top5_seul", "alpha",
                    "diag_alpha", "diag_bas", "diag_haut", "propres_alpha", "minutes"])

    if step == 0:
        r = verrou(modele, diag, prop, pad, dev)
        print(f"[{paris()}] pas 0 (tete nulle : toutes notes egales -> premier candidat) "
              f"diag {r['diag'][0]:.2%} (alpha {r['alpha']})", flush=True)

    modele.train(); t0 = time.time(); mi = step * a.accum; acc = 0.0; s0 = step; dern = float("nan")
    sautes = 0
    while step < total:
        for g, lr0 in zip(opt.param_groups, base_lr):
            g["lr"] = lr0 * lr_a(step)
        for _ in range(a.accum):
            b = micro[mi % len(micro)]; mi += 1
            x, c, m = tenseurs(b, T, C, O, pad, dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                l = perte(modele(x), c, m) / a.accum
            if not torch.isfinite(l):
                continue                                   # micro-lot saute (compte plus bas)
            l.backward(); acc += l.item()
        # GARDE (25/09) : v2 a diverge au pas ~1100, un seul pas non fini a mis
        # des NaN dans tous les poids ; clip_grad_norm_ ne protege pas d'un NaN.
        norme = torch.nn.utils.clip_grad_norm_(modele.parameters(), a.clip)
        if torch.isfinite(norme):
            opt.step()
        else:
            sautes += 1
            print(f"[{paris()}] pas {step} SAUTE : gradient non fini ({sautes} au total)", flush=True)
        opt.zero_grad(set_to_none=True); step += 1
        if step % 25 == 0:
            el = time.time() - t0
            print(f"[{paris()}] pas {step}/{total} perte {acc/25:.4f} lr {opt.param_groups[0]['lr']:.2e} "
                  f"| {el/60:.1f} min, reste ~{el/max(1, step - s0)*(total-step)/60:.0f} min",
                  flush=True)
            dern = acc / 25; acc = 0.0
        if step % a.eval_every == 0 or step == total:
            pv = val(); r = verrou(modele, diag, prop, pad, dev)
            d = r["diag"]
            print(f"[{paris()}] EVAL pas {step} : perte val {pv:.4f} | verrou diag top-5 seul "
                  f"{r['seul_top5']:.2%}, top-2 {r['top2']:.2%} | alpha {r['alpha']} -> "
                  f"{d[0]:.2%} [{d[1]:.2%} ; {d[2]:.2%}] (bot {r['propres_alpha']:.2%})", flush=True)
            w.writerow([step, opt.param_groups[0]["lr"], dern, pv, r["seul_top5"], r["alpha"],
                        d[0], d[1], d[2], r["propres_alpha"], (time.time() - t0) / 60]); fcsv.flush()
            if not math.isfinite(pv):
                print(f"[{paris()}] perte de validation non finie : aucune sauvegarde, arret", flush=True)
                return
            etat = {"modele": modele.state_dict(), "model_config": ck["model_config"], "step": step,
                    "best": best, "verrou": r, "perte_val": pv}
            if r["propres_alpha"] < best:        # sélection sur les positions du bot, jamais sur le diagnostic
                best = etat["best"] = r["propres_alpha"]
                torch.save(etat, a.out)
            etat["opt"] = opt.state_dict()
            torch.save(etat, a.out.replace(".pt", "_etat.pt"))
    print(f"[{paris()}] TERMINE", flush=True)


if __name__ == "__main__":
    main()
