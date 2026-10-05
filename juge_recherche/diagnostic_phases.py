"""Où le bot perd-il ses parties ?

Cinq jours d'optimisation sans jamais regarder les symptômes. On a 3 300 parties
classées réelles sur le disque : on les passe à Stockfish pour localiser la
perte d'Elo.

Trois mesures, dans l'ordre d'utilité.

**La perte moyenne en centipions par coup (ACPL)**, ventilée par phase. C'est la
mesure standard de la qualité de jeu, et elle dit dans quelle phase le modèle
est le plus faible **relativement**.

**Le taux de gaffes**, un coup qui fait chuter l'évaluation de plus de 200
centipions, par phase. On a établi que l'Elo est gouverné par le taux de
gaffes, pas par la qualité moyenne.

**La forme de la défaite** : une gaffe unique et fatale, ou une dégradation
lente ? Les deux appellent des correctifs opposés.

Usage :
    python diagnostic_phases.py --n-defaites 800 --n-victoires 800
"""

import argparse
import os
import json
import multiprocessing as mp
import time
from collections import defaultdict

import chess
import chess.engine

SF = None
NOEUDS = 20000


def _init(chemin, noeuds):
    global SF, NOEUDS
    NOEUDS = noeuds
    SF = chess.engine.SimpleEngine.popen_uci(chemin)
    SF.configure({"Skill Level": 20, "Threads": 1, "Hash": 32})


def phase(board, ply):
    """Phase par le materiel restant, plus robuste que le numero de coup."""
    n = len(board.piece_map())
    if ply < 20:
        return "ouverture"
    return "finale" if n <= 12 else "milieu"


def _traiter(args):
    ligne, moi = args
    g = json.loads(ligne)
    p = g["players"]
    blanc = p["white"].get("user", {}).get("id") == moi
    w = g.get("winner")
    issue = "N" if w is None else ("V" if (w == "white") == blanc else "D")
    board = chess.Board()
    lim = chess.engine.Limit(nodes=NOEUDS)
    pertes = []          # (phase, perte_en_cp, ply)
    try:
        prec = SF.analyse(board, lim)["score"].white().score(mate_score=10000)
    except Exception:
        return None
    for ply, san in enumerate(g.get("moves", "").split()):
        try:
            mv = board.parse_san(san)
        except Exception:
            break
        a_moi = (board.turn == chess.WHITE) == blanc
        ph = phase(board, ply)
        board.push(mv)
        try:
            ap = SF.analyse(board, lim)["score"].white().score(mate_score=10000)
        except Exception:
            break
        if a_moi:
            # perte du point de vue du bot : positive = il a degrade sa position
            perte = (prec - ap) if blanc else (ap - prec)
            pertes.append((ph, max(0, perte), ply))
        prec = ap
    return issue, pertes


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--parties", default="diag/parties.ndjson")
    p.add_argument("--stockfish",
                   default=os.environ.get("STOCKFISH", "stockfish"))
    p.add_argument("--moi", default="philidor-142m")
    p.add_argument("--workers", type=int, default=10)
    p.add_argument("--noeuds", type=int, default=20000)
    p.add_argument("--n-defaites", type=int, default=800)
    p.add_argument("--n-victoires", type=int, default=800)
    p.add_argument("--out", default="diag/diagnostic_phases.json")
    args = p.parse_args()

    lignes = [l for l in open(args.parties) if l.strip()]
    defs, vics = [], []
    for l in lignes:
        g = json.loads(l)
        blanc = g["players"]["white"].get("user", {}).get("id") == args.moi
        w = g.get("winner")
        if w is None:
            continue
        gagne = (w == "white") == blanc
        if gagne and len(vics) < args.n_victoires:
            vics.append((l, args.moi))
        elif not gagne and len(defs) < args.n_defaites:
            defs.append((l, args.moi))
    taches = defs + vics
    print(f"[diagnostic] {len(defs)} defaites + {len(vics)} victoires, "
          f"Stockfish a {args.noeuds:,} noeuds, {args.workers} ouvriers",
          flush=True)

    acpl = defaultdict(list)          # (issue, phase) -> pertes
    gaffes = defaultdict(lambda: [0, 0])   # (issue, phase) -> [gaffes, coups]
    forme = defaultdict(int)
    t0 = time.perf_counter()
    n = 0
    with mp.Pool(args.workers, initializer=_init,
                 initargs=(args.stockfish, args.noeuds)) as pool:
        for r in pool.imap_unordered(_traiter, taches, chunksize=4):
            if not r:
                continue
            issue, pertes = r
            n += 1
            if not pertes:
                continue
            for ph, pe, _ in pertes:
                acpl[(issue, ph)].append(pe)
                gaffes[(issue, ph)][1] += 1
                if pe >= 200:
                    gaffes[(issue, ph)][0] += 1
            if issue == "D":
                pire = max(pe for _, pe, _ in pertes)
                forme["une gaffe fatale (>400 cp)" if pire >= 400 else
                      ("une erreur nette (200-400)" if pire >= 200 else
                       "degradation lente (<200)")] += 1
            if n % 200 == 0:
                print(f"  {n}/{len(taches)} parties, "
                      f"{(time.perf_counter()-t0)/60:.1f} min", flush=True)

    def med(t):
        return sorted(t)[len(t)//2] if t else 0

    print(f"\n=== PERTE MOYENNE PAR COUP (centipions) ===")
    print(f"{'phase':>12}{'defaites':>12}{'victoires':>12}{'coups D':>10}")
    for ph in ("ouverture", "milieu", "finale"):
        d, v = acpl[("D", ph)], acpl[("V", ph)]
        if d or v:
            print(f"{ph:>12}{(sum(d)/len(d) if d else 0):>12.0f}"
                  f"{(sum(v)/len(v) if v else 0):>12.0f}{len(d):>10}")
    print(f"\n=== TAUX DE GAFFES (perte >= 200 cp) ===")
    print(f"{'phase':>12}{'defaites':>12}{'victoires':>12}")
    for ph in ("ouverture", "milieu", "finale"):
        gd, td = gaffes[("D", ph)]
        gv, tv = gaffes[("V", ph)]
        if td or tv:
            print(f"{ph:>12}{(gd/td if td else 0):>11.1%}{(gv/tv if tv else 0):>12.1%}")
    print(f"\n=== FORME DE LA DEFAITE ===")
    tot = sum(forme.values())
    for k, v in sorted(forme.items(), key=lambda x: -x[1]):
        print(f"  {k:32} {v:5}  ({v/max(tot,1):5.1%})")
    print(f"\nduree : {(time.perf_counter()-t0)/60:.1f} min")

    json.dump({"acpl": {f"{k[0]}_{k[1]}": (sum(v)/len(v) if v else 0)
                        for k, v in acpl.items()},
               "gaffes": {f"{k[0]}_{k[1]}": v for k, v in gaffes.items()},
               "forme": dict(forme)}, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
