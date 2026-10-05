"""Étape B3 : la recherche comme maître (la boucle d'AlphaZero, version minimale).

B1 et B2 apprenaient à la politique le choix du sélecteur « politique +
valeur » : gain modeste (-1,2 point hors ligne), rien de visible en duel. Ici la
cible est la RÉPARTITION DES VISITES d'une recherche à 64 simulations (valeur
v2), un maître bien plus fort (10,98 % de gaffes sur le diagnostic contre
14,30 % pour le sélecteur).

Positions : jusqu'à `--par-partie` demi-coups tirés au hasard (>= ply-min) dans
chaque partie du corpus de run2 (validation sautée). Recherches avancées en
lots (recherche.chercher). Même format de sortie que iteration_etiqueter.py,
avec lv = log(visites) : iteration_train.py --alpha 0 --T 1 donne pi ∝ visites.
"""
import argparse, json, math, os, random, time
import numpy as np
import torch
import chess
from model import ChessGPT, ModelConfig
from juge_recherche.valeur_train import Valeur
from juge_recherche.recherche import Recherche, chercher


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--games", default="data/games_uci.txt")
    p.add_argument("--val", default="data/val_games.txt")
    p.add_argument("--debut", type=int, default=200_000, help="ligne de départ (après les parties de B1)")
    p.add_argument("--n-positions", type=int, default=100_000)
    p.add_argument("--par-partie", type=int, default=5)
    p.add_argument("--ply-min", type=int, default=8)
    p.add_argument("--sims", type=int, default=64)
    p.add_argument("--cpuct", type=float, default=0.1)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--valeur", default="checkpoints/valeur.pt")
    p.add_argument("--politique", default="checkpoints/run2_best.pt")
    p.add_argument("--out", default="data/iteration/b3")
    p.add_argument("--par", type=int, default=128)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--device", default="cuda:0")
    a = p.parse_args()
    dev = a.device
    rng = random.Random(a.seed)

    v = json.load(open("data/vocab.json")); stoi, bos, pad = v["stoi"], v["bos_id"], v["pad_id"]
    ck = torch.load(a.politique, map_location="cpu", weights_only=False)
    pol = ChessGPT(ModelConfig(**ck["model_config"])); pol.load_state_dict(ck["model"]); pol = pol.to(dev).eval()
    cv = torch.load(a.valeur, map_location="cpu", weights_only=False)
    val = Valeur(ChessGPT(ModelConfig(**cv["model_config"]))); val.load_state_dict(cv["modele"])
    val = val.to(dev).eval()
    ctx = pol.cfg.block_size

    parties_val = {l.strip() for l in open(a.val)}
    os.makedirs(a.out, exist_ok=True)
    G, GO, POS, CAND, LV, LP = [], [0], [], [], [], []
    t0 = time.time()

    def positions():
        with open(a.games) as f:
            for i, l in enumerate(f):
                if i < a.debut:
                    continue
                s = l.strip()
                if s in parties_val:
                    continue
                u = s.split()[:ctx - 1]
                if len(u) < a.ply_min + 10 or any(m not in stoi for m in u):
                    continue
                gi = len(GO) - 1
                ids = [bos] + [stoi[m] for m in u]
                G.append(np.asarray(ids, np.uint16)); GO.append(GO[-1] + len(ids))
                for t in sorted(rng.sample(range(a.ply_min, len(u)), min(a.par_partie, len(u) - a.ply_min))):
                    b = chess.Board()
                    for m in u[:t]:
                        b.push_uci(m)
                    if b.is_game_over() or b.legal_moves.count() < 2:
                        continue
                    yield gi, t, b, ids[:t + 1]

    lot = []

    def traiter(lot):
        R = [Recherche(b, ids, stoi, k=a.k, cpuct=a.cpuct, fpu=0.0, ctx=ctx) for _, _, b, ids in lot]
        chercher(R, pol, val, a.sims, pad, dev, lot=8)
        for (gi, t, _, _), r in zip(lot, R):
            en = r.racine.enfants
            c5 = [e.tok for e in en] + [pad] * (a.k - len(en))
            v5 = [math.log(e.n) if e.n > 0 else -1e4 for e in en] + [-1e4] * (a.k - len(en))
            l5 = [math.log(max(e.p, 1e-9)) for e in en] + [-1e4] * (a.k - len(en))
            POS.append((gi, t)); CAND.append(c5); LV.append(v5); LP.append(l5)

    for item in positions():
        lot.append(item)
        if len(lot) == a.par:
            traiter(lot); lot = []
            if len(POS) % 5120 < a.par:
                dt = time.time() - t0
                print(f"  {len(POS):,} positions ({dt/60:.1f} min, {dt/len(POS)*1000:.0f} ms/position)", flush=True)
        if len(POS) + len(lot) >= a.n_positions:
            break
    if lot:
        traiter(lot)

    np.concatenate(G).tofile(f"{a.out}/parties.u16")
    np.asarray(GO, np.int64).tofile(f"{a.out}/parties_off.i64")
    np.asarray(POS, np.int32).tofile(f"{a.out}/pos.i32")
    np.asarray(CAND, np.uint16).tofile(f"{a.out}/cand.u16")
    np.asarray(LV, np.float16).tofile(f"{a.out}/lv.f16")
    np.asarray(LP, np.float16).tofile(f"{a.out}/lp.f16")
    json.dump({"k": a.k, "n_parties": len(GO) - 1, "n_positions": len(POS), "ply_min": a.ply_min,
               "debut": a.debut, "sims": a.sims, "cpuct": a.cpuct, "valeur": a.valeur,
               "cible": "log(visites) de la racine"}, open(f"{a.out}/info.json", "w"))
    print(f"[b3] {len(POS):,} positions de {len(GO)-1:,} parties | {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
