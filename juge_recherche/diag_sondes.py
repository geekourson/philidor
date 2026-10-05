"""Mesure 3 : le modèle représente-t-il les attaques ailleurs que sur le roi ?

Hypothèse testée : le modèle calcule les attaques sur le ROI parce que la
légalité l'exige (un coup qui laisse son roi en échec est illégal, et le modèle
produit 98,85 % de coups légaux sans masque), mais pas sur les autres pièces.
Si c'est vrai, il ne « voit » pas ses pièces en prise, ce qui expliquerait des
gaffes de profondeur 1.

Trois cibles, toutes du point de vue du joueur AU TRAIT :
  (a) par case : une pièce alliée occupe X et est attaquée par l'adversaire ;
  (b) par case : une pièce alliée NON-ROI occupe X, est attaquée et n'est pas
      défendue, « en prise » ;
  (c) globale  : le joueur au trait est en échec. C'est le CONTRÔLE POSITIF ;
      la légalité l'exige, la sonde doit être proche de 100 %.

Sondes linéaires, une régression logistique par case et par couche. Elles sont
entraînées ensemble sous forme d'une couche `Linear(d, 64)` à perte BCE : chaque
sortie a sa propre ligne de poids, c'est donc bien 64 régressions indépendantes.

Jeu de test séparé, jamais vu par la sonde. Et un témoin : la même sonde sur un
modèle aux POIDS ALÉATOIRES, qui donne le niveau atteignable sans aucune
connaissance apprise.

Sur des cibles rares, la justesse brute trompe : une sonde qui répond toujours
« non » a déjà 95 % de justesse. On publie donc aussi l'AUC et le taux de
positifs.

    python diag_sondes.py --n 20000 --device cuda:0
"""

import argparse
import json
import os
import time

import chess
import numpy as np
import torch
import torch.nn as nn

from model import ChessGPT, ModelConfig


def cibles(board):
    """(a) attaquee, (b) en prise, (c) echec, du point de vue du trait."""
    moi, adv = board.turn, not board.turn
    a = np.zeros(64, dtype=np.float32)
    b = np.zeros(64, dtype=np.float32)
    for sq, pc in board.piece_map().items():
        if pc.color != moi:
            continue
        attaquee = board.is_attacked_by(adv, sq)
        if attaquee:
            a[sq] = 1.0
            if pc.piece_type != chess.KING and not board.is_attacked_by(moi, sq):
                b[sq] = 1.0
    return a, b, np.float32(board.is_check())


def auc(y, s):
    """Aire sous la courbe ROC, par les rangs. nan si une seule classe."""
    y = np.asarray(y); s = np.asarray(s)
    npos, nneg = y.sum(), (1 - y).sum()
    if npos == 0 or nneg == 0:
        return float("nan")
    r = np.empty(len(s)); r[np.argsort(s)] = np.arange(1, len(s) + 1)
    return float((r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def extraire(model, cfg, lignes, stoi, bos, device, couches, lot=64):
    """Activations au DERNIER jeton, pour chaque couche demandee."""
    etat = {}
    hooks = []
    for nom, mod in couches.items():
        hooks.append(mod.register_forward_hook(
            lambda m, i, o, n=nom: etat.__setitem__(n, o)))
    H = {n: [] for n in couches}
    A, B, C = [], [], []
    with torch.no_grad():
        # Une position a la fois : le bourrage a gauche par <bos> mettait le
        # modele hors distribution, faute de masque d'attention. Meme bug que
        # dans diag_modele.py, meme correction.
        for k in range(len(lignes)):
            lot_d = [json.loads(lignes[k])]
            ids = [bos] + [stoi[m] for m in lot_d[0]["prefixe"] if m in stoi]
            x = torch.tensor([ids[-cfg.block_size:]], dtype=torch.long)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                model(x.to(device))
            for n in couches:
                H[n].append(etat[n][:, -1, :].float().cpu().numpy())
            for d in lot_d:
                a, b, c = cibles(chess.Board(d["fen"]))
                A.append(a); B.append(b); C.append(c)
    for h in hooks:
        h.remove()
    return ({n: np.concatenate(v) for n, v in H.items()},
            np.array(A), np.array(B), np.array(C, dtype=np.float32))


def sonde(Xtr, Ytr, Xte, Yte, device, epochs=2000, lr=3e-3):
    """Regressions logistiques independantes, une par colonne de Y.

    2 000 epoques et non 200 : a 200 la sonde n'a PAS converge. Verifie sur le
    controle positif, ou la justesse passe de 0,877 a 0,954 et l'AUC de 0,871 a
    0,911 entre 200 et 2 000 epoques, puis plafonne. Publier les chiffres a 200
    aurait sous-estime tout le monde de la meme facon, mais aurait surtout fait
    conclure a tort que le modele ne represente pas l'echec."""
    d, k = Xtr.shape[1], Ytr.shape[1] if Ytr.ndim > 1 else 1
    Ytr2 = Ytr.reshape(len(Ytr), k); Yte2 = Yte.reshape(len(Yte), k)
    mu, sd = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-6
    T = lambda z: torch.tensor((z - mu) / sd, dtype=torch.float32, device=device)
    xtr, xte = T(Xtr), T(Xte)
    ytr = torch.tensor(Ytr2, dtype=torch.float32, device=device)
    lin = nn.Linear(d, k).to(device)
    opt = torch.optim.AdamW(lin.parameters(), lr=lr, weight_decay=1e-3)
    lossf = nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        opt.zero_grad(); lossf(lin(xtr), ytr).backward(); opt.step()
    with torch.no_grad():
        s = lin(xte).cpu().numpy()
    just, aucs, taux = [], [], []
    for j in range(k):
        y = Yte2[:, j]
        if y.sum() == 0:
            continue
        just.append(float(((s[:, j] > 0) == (y > 0.5)).mean()))
        aucs.append(auc(y, s[:, j]))
        taux.append(float(y.mean()))
    return (float(np.mean(just)), float(np.nanmean(aucs)),
            float(np.mean(taux)), len(just))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/run2_best.pt")
    p.add_argument("--vocab", default="data/vocab.json")
    p.add_argument("--positions", default="diag/sondes_positions.jsonl")
    p.add_argument("--out", default="diag/sondes.json")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--n", type=int, default=20000)
    p.add_argument("--test-frac", type=float, default=0.25)
    args = p.parse_args()

    with open(args.vocab) as f:
        v = json.load(f)
    stoi, bos = v["stoi"], v["bos_id"]
    lignes = open(args.positions).read().splitlines()[:args.n]
    print(f"[sondes] {len(lignes):,} positions", flush=True)

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ModelConfig(**ck["model_config"])
    res = {}
    t0 = time.time()
    for nom_modele, aleatoire in (("run2", False), ("temoin aleatoire", True)):
        m = ChessGPT(cfg)
        if not aleatoire:
            m.load_state_dict(ck["model"])
        m = m.to(args.device).eval()
        mi, mj = cfg.n_layer // 2, int(cfg.n_layer * 0.75)
        couches = {f"bloc{mi}": m.blocks[mi - 1], f"bloc{mj}": m.blocks[mj - 1],
                   "finale": m.norm_final}
        H, A, B, C = extraire(m, cfg, lignes, stoi, bos, args.device, couches)
        n_te = int(len(A) * args.test_frac)
        print(f"  [{nom_modele}] extraction faite ({time.time()-t0:.0f}s), "
              f"test sur {n_te:,}", flush=True)
        for couche, X in H.items():
            Xtr, Xte = X[n_te:], X[:n_te]
            for cle_c, Y in (("a_attaquee", A), ("b_en_prise", B),
                             ("c_echec", C)):
                j, au, tx, nc = sonde(Xtr, Y[n_te:], Xte, Y[:n_te], args.device)
                res.setdefault(nom_modele, {}).setdefault(couche, {})[cle_c] = {
                    "justesse": round(j, 4), "auc": round(au, 4),
                    "taux_positifs": round(tx, 4), "cases": nc}
                print(f"    {couche:<8} {cle_c:<12} justesse {j:.3f} | "
                      f"AUC {au:.3f} | positifs {tx:.1%}", flush=True)
        del m
        torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=2)
    print(f"\n-> {args.out}  ({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
