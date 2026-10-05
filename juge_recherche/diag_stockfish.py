"""Évaluation Stockfish de TOUS les coups légaux, profondeur 18.

C'est la référence contre laquelle tout le diagnostic se mesure. Pour chaque
position on obtient un dictionnaire coup UCI -> centipions.

Convention de signe, et c'est le piège classique. `info["score"]` est un
`PovScore` ; `.relative` le ramène au point de vue du joueur AU TRAIT à la
racine. Un coup vaut donc d'autant plus que son cp est grand, quelle que soit la
couleur. Une gaffe est alors simplement :

    meilleur_cp - cp_du_coup_du_modele >= 100

sans aucune inversion à faire pour les Noirs.

Les mats sont bornés à ±10 000 : les laisser à l'infini rendrait tout écart
incalculable.

    python diag_stockfish.py --workers 10 --profondeur 18
"""

import argparse
import json
import multiprocessing as mp
import os
import time

import chess
import chess.engine

SF_BIN = os.environ.get("STOCKFISH", "stockfish")
SF = None
PROF = 18


def _init(chemin, prof, hash_mo):
    global SF, PROF
    PROF = prof
    SF = chess.engine.SimpleEngine.popen_uci(chemin)
    SF.configure({"Threads": 1, "Hash": hash_mo})


def _traiter(ligne):
    d = json.loads(ligne)
    b = chess.Board(d["fen"])
    n = b.legal_moves.count()
    if n == 0:
        return None
    try:
        infos = SF.analyse(b, chess.engine.Limit(depth=PROF), multipv=n)
    except chess.engine.EngineError:
        return None
    cps = {}
    for info in infos:
        pv = info.get("pv")
        if not pv:
            continue
        cps[pv[0].uci()] = info["score"].relative.score(mate_score=10_000)
    if len(cps) < n:
        # MultiPV peut rendre moins de lignes que demande sur des positions
        # forcees ; on garde, mais on le signale pour pouvoir l'ecarter apres.
        pass
    return json.dumps({"fen": d["fen"], "cp": cps, "n_legaux": n},
                      separators=(",", ":"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--positions", default="diag/positions.jsonl")
    p.add_argument("--out", default="diag/stockfish18.jsonl")
    p.add_argument("--stockfish", default=SF_BIN)
    p.add_argument("--profondeur", type=int, default=18)
    p.add_argument("--workers", type=int, default=10)
    p.add_argument("--hash", type=int, default=128)
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    lignes = open(args.positions).read().splitlines()
    print(f"[sf] {len(lignes):,} positions, profondeur {args.profondeur}, "
          f"{args.workers} workers", flush=True)

    t0 = time.time()
    n = manquants = 0
    with open(args.out, "w") as g, mp.Pool(
            args.workers, initializer=_init,
            initargs=(args.stockfish, args.profondeur, args.hash)) as pool:
        for res in pool.imap_unordered(_traiter, lignes, chunksize=4):
            if not res:
                continue
            d = json.loads(res)
            if len(d["cp"]) < d["n_legaux"]:
                manquants += 1
            g.write(res + "\n")
            n += 1
            if n % 250 == 0:
                dt = time.time() - t0
                g.flush()
                print(f"  {n:,}/{len(lignes):,} | {dt/60:.1f} min "
                      f"({n/dt:.2f} pos/s, reste "
                      f"{(len(lignes)-n)/(n/dt)/3600:.1f} h)", flush=True)

    dt = time.time() - t0
    print(f"\n{n:,} positions evaluees -> {args.out}")
    print(f"  lignes MultiPV incompletes : {manquants:,}")
    print(f"duree : {dt/3600:.2f} h ({n/dt:.2f} positions/s)")


if __name__ == "__main__":
    main()
