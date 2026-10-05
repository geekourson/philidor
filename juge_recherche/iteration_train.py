"""Étape B, affinage : la politique apprend à jouer le coup reclassé.

Cible à chaque position étiquetée (iteration_etiqueter.py) : une distribution
sur les k candidats de la politique,
    pi(c) ∝ exp((logit_valeur(c) + alpha · log p(c)) / T),
dont l'argmax est exactement le choix du moteur « politique + valeur » (alpha
0,25). Perte : entropie croisée entre pi et la politique, sur tout le
vocabulaire. Avant `ply-min` (ouverture, non étiquetée) : le coup humain,
comme à l'entraînement de run2, pour ne pas désapprendre les ouvertures.

Différence avec l'essai « coups de Stockfish » (−35,9 Elo le 11/09) : la cible
n'est pas un coup extérieur, imprévisible sans calcul, mais un reclassement de
ses propres cinq premiers coups. La politique n'a qu'à redistribuer sa masse.

Verrou intégré : argmax de la politique affinée sur les positions du bot
(sélection du point de sauvegarde) et du diagnostic (publication).
Repères diagnostic : argmax run2 18,32 % | politique + valeur 15,28 %.
"""
import argparse, csv, json, math, os, time
import numpy as np
import torch
import torch.nn.functional as F
from model import ChessGPT, ModelConfig
from juge_recherche.recherche import tronc
from juge_recherche.recherche_verrou import positions, mesures, wilson, paris


@torch.no_grad()
def argmax_positions(pol, X, stoi, pad, dev, lot=256):
    choix = []
    for i in range(0, len(X), lot):
        b = X[i:i + lot]
        seqs = [x["ids"][-pol.cfg.block_size:] for x in b]
        L = max(len(s) for s in seqs)
        xx = torch.full((len(b), L), pad, dtype=torch.long)
        for r, s in enumerate(seqs):
            xx[r, :len(s)] = torch.tensor(s)
        fin = torch.tensor([len(s) - 1 for s in seqs], device=dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg = pol.lm_head(tronc(pol, xx.to(dev))[torch.arange(len(b), device=dev), fin]).float().cpu()
        for r, x in enumerate(b):
            leg = [m.uci() for m in x["b"].legal_moves]
            choix.append(max(leg, key=lambda u: lg[r, stoi[u]].item()))
    return choix


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/iteration/b1")
    p.add_argument("--ckpt", default="checkpoints/run2_best.pt")
    p.add_argument("--out", default="checkpoints/b1_politique.pt")
    p.add_argument("--log", default="logs/distillation.csv")
    p.add_argument("--alpha", type=float, default=0.25)
    p.add_argument("--T", type=float, default=1.0)
    p.add_argument("--lambda-humain", type=float, default=1.0, help="poids du coup humain avant ply-min")
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--lr-min-frac", type=float, default=0.1)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--wd", type=float, default=0.1)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--epoques", type=float, default=1.0)
    p.add_argument("--eval-every", type=int, default=200)
    p.add_argument("--val-frac", type=float, default=0.02)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--device", default="cuda:0")
    a = p.parse_args()
    dev = a.device
    torch.manual_seed(a.seed)

    v = json.load(open("data/vocab.json")); stoi, bos, pad = v["stoi"], v["bos_id"], v["pad_id"]
    info = json.load(open(f"{a.data}/info.json")); k = info["k"]; ply_min = info["ply_min"]
    G = np.fromfile(f"{a.data}/parties.u16", np.uint16)
    GO = np.fromfile(f"{a.data}/parties_off.i64", np.int64)
    POS = np.fromfile(f"{a.data}/pos.i32", np.int32).reshape(-1, 2)
    C = np.fromfile(f"{a.data}/cand.u16", np.uint16).reshape(-1, k).astype(np.int64)
    LV = np.fromfile(f"{a.data}/lv.f16", np.float16).reshape(-1, k).astype(np.float32)
    LP = np.fromfile(f"{a.data}/lp.f16", np.float16).reshape(-1, k).astype(np.float32)
    n_g = len(GO) - 1
    # cible pi, et index des positions par partie
    s = (LV + a.alpha * LP) / a.T
    s[C == pad] = -1e9
    PI = torch.softmax(torch.from_numpy(s), -1).numpy()
    par_partie = [[] for _ in range(n_g)]
    for i, (g, t) in enumerate(POS):
        par_partie[g].append(i)
    rng = np.random.default_rng(a.seed)
    ordre = rng.permutation(n_g); n_val = int(n_g * a.val_frac)
    iva, itr = ordre[:n_val], ordre[n_val:]
    garde = float((C[:, 0] == C[np.arange(len(C)), PI.argmax(1)]).mean())
    print(f"[{paris()}] {n_g:,} parties ({len(itr):,} entrainement), {len(POS):,} positions etiquetees | "
          f"le selecteur garde le top-1 de la politique dans {garde:.1%} des positions", flush=True)

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = ModelConfig(**ck["model_config"])
    pol = ChessGPT(cfg); pol.load_state_dict(ck["model"]); pol = pol.to(dev)

    fd = ("diag/positions.jsonl", "diag/modele.jsonl", "diag/stockfish18.jsonl")
    fp = ("diag/propres_positions.jsonl", "diag/propres_modele.jsonl", "diag/propres_sf18.jsonl")
    XD, XP = positions(*fd, stoi, bos), positions(*fp, stoi, bos)

    def verrou():
        pol.eval()
        md = mesures(XD, argmax_positions(pol, XD, stoi, pad, dev))
        mp_ = mesures(XP, argmax_positions(pol, XP, stoi, pad, dev))
        pol.train()
        return md, mp_

    def lot_tenseurs(gs):
        L = max(int(GO[g + 1] - GO[g]) for g in gs)
        x = torch.full((len(gs), L), pad, dtype=torch.long)
        yh = torch.full((len(gs), L), -1, dtype=torch.long)       # coup humain (ouverture)
        idx_b, idx_t, cand, pi = [], [], [], []
        for r, g in enumerate(gs):
            seq = torch.from_numpy(G[GO[g]:GO[g + 1]].astype(np.int64))
            n = len(seq); x[r, :n] = seq
            tmax = min(ply_min, n - 1)
            yh[r, :tmax] = seq[1:tmax + 1]
            for i in par_partie[g]:
                idx_b.append(r); idx_t.append(int(POS[i, 1])); cand.append(C[i]); pi.append(PI[i])
        return (x.to(dev), yh.to(dev), torch.tensor(idx_b, device=dev), torch.tensor(idx_t, device=dev),
                torch.from_numpy(np.stack(cand)).to(dev), torch.from_numpy(np.stack(pi)).to(dev))

    def perte(gs):
        x, yh, ib, it, cand, pi = lot_tenseurs(gs)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg = pol(x)[0].float()
        lsm = torch.log_softmax(lg[ib, it], -1)                   # (n, V)
        lb = -(pi * lsm.gather(1, cand)).sum(1).mean()
        acc = (lg[ib, it].argmax(-1) == cand.gather(1, pi.argmax(1, keepdim=True)).squeeze(1)).float().mean()
        lh = F.cross_entropy(lg.view(-1, lg.size(-1)), yh.view(-1), ignore_index=-1)
        return lb + a.lambda_humain * lh, lb, acc

    dec = [q for q in pol.parameters() if q.dim() >= 2]; nodec = [q for q in pol.parameters() if q.dim() < 2]
    opt = torch.optim.AdamW([{"params": dec, "weight_decay": a.wd}, {"params": nodec, "weight_decay": 0.0}],
                            lr=a.lr, betas=(0.9, 0.95), fused=True)
    total = int(len(itr) * a.epoques) // a.batch

    def lr_a(st):
        if st < a.warmup:
            return (st + 1) / a.warmup
        f = (st - a.warmup) / max(1, total - a.warmup)
        return a.lr_min_frac + (1 - a.lr_min_frac) * 0.5 * (1 + math.cos(math.pi * min(1.0, f)))

    os.makedirs(os.path.dirname(a.log), exist_ok=True); os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fcsv = open(a.log, "w", newline=""); w = csv.writer(fcsv)
    w.writerow(["step", "perte_b_val", "accord_val", "diag_g100", "diag_moy", "bot_g100", "bot_moy"])
    md, mp_ = verrou()
    print(f"[{paris()}] pas 0 : diag {md['g100']:.2%} (moy {md['moy']:.1f}) | bot {mp_['g100']:.2%}", flush=True)
    best = mp_["g100"]
    pol.train(); t0 = time.time(); st = 0; ep_ordre = itr.copy()
    while st < total:
        rng.shuffle(ep_ordre)
        for j in range(0, len(ep_ordre) - a.batch + 1, a.batch):
            if st >= total:
                break
            for gr in opt.param_groups:
                gr["lr"] = a.lr * lr_a(st)
            l, lb, acc = perte(ep_ordre[j:j + a.batch])
            l.backward(); torch.nn.utils.clip_grad_norm_(pol.parameters(), 1.0)
            opt.step(); opt.zero_grad(set_to_none=True); st += 1
            if st % 25 == 0:
                print(f"[{paris()}] pas {st}/{total} perte B {lb.item():.4f} accord {acc.item():.1%} "
                      f"| {(time.time()-t0)/60:.1f} min", flush=True)
            if st % a.eval_every == 0 or st == total:
                pol.eval()
                with torch.no_grad():
                    vl = va = 0.0; nb = 0
                    for jj in range(0, len(iva), a.batch):
                        _, lb2, acc2 = perte(iva[jj:jj + a.batch]); vl += lb2.item(); va += acc2.item(); nb += 1
                md, mp_ = verrou()
                lo, hi = wilson(md["g100"], md["n"])
                print(f"[{paris()}] EVAL pas {st} : perte B val {vl/nb:.4f}, accord {va/nb:.1%} | argmax affine : "
                      f"diag {md['g100']:.2%} [{lo:.2%} ; {hi:.2%}] moy {md['moy']:.1f} | bot {mp_['g100']:.2%} "
                      f"moy {mp_['moy']:.1f}", flush=True)
                w.writerow([st, vl / nb, va / nb, md["g100"], md["moy"], mp_["g100"], mp_["moy"]]); fcsv.flush()
                if mp_["g100"] < best:                  # sélection sur le bot, jamais sur le diagnostic
                    best = mp_["g100"]
                    torch.save({"model": pol.state_dict(), "model_config": ck["model_config"], "step": st,
                                "diag": md, "bot": mp_}, a.out)
                pol.train()
    print(f"[{paris()}] TERMINE", flush=True)


if __name__ == "__main__":
    main()
