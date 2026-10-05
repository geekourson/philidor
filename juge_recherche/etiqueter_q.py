"""Étiquetage Stockfish des coups CANDIDATS du modèle : les cibles Q.

V(s) note une position. Q(s,a) note un coup dans une position. C'est Q qu'il
nous faut : le modèle doit apprendre que SES mauvais coups sont mauvais, or il
ne les joue jamais dans le corpus humain, donc aucune étiquette ne les couvre.

q_candidats.py a produit, pour chaque position, les 3 coups que le modèle
propose. Ici on demande à Stockfish de noter exactement ces 3 coups.

Décision : une seule recherche par position avec `root_moves` + `multipv`,
plutôt que trois recherches sur les trois positions filles. Stockfish partage
alors son arbre et sa table de transposition entre les trois branches, ce qui
est nettement moins cher que trois recherches indépendantes, et les trois
scores sortent sur la même échelle (celle du trait au noeud racine).

Convention : le cp renvoyé est `score.relative`, donc du point de vue du
joueur au trait. Positif = bon pour celui qui joue le coup.

Usage :
    python etiqueter_q.py --in .../candidats.jsonl --out .../q_labels.jsonl
"""

import argparse
import os
import json
import multiprocessing as mp
import time

import chess
import chess.engine

SF = None
NOEUDS = 200_000


def init(chemin, noeuds):
    global SF, NOEUDS
    NOEUDS = noeuds
    SF = chess.engine.SimpleEngine.popen_uci(chemin)
    SF.configure({"Threads": 1, "Hash": 64})


def traiter(ligne):
    global SF
    try:
        d = json.loads(ligne)
    except json.JSONDecodeError:
        return None
    board = chess.Board()
    try:
        for u in d["prefixe"]:
            board.push_uci(u)
    except (ValueError, AssertionError):
        return None
    if board.is_game_over():
        return None

    coups = []
    for u in d["candidats_modele"]:
        try:
            mv = chess.Move.from_uci(u)
        except ValueError:
            continue
        if mv in board.legal_moves:
            coups.append(mv)
    if not coups:
        return None

    try:
        infos = SF.analyse(board, chess.engine.Limit(nodes=NOEUDS),
                           multipv=len(coups), root_moves=coups)
    except chess.engine.EngineError:
        return None

    q = {}
    for info in infos:
        pv = info.get("pv")
        if not pv:
            continue
        # borné : un mat vaut plus que n'importe quelle évaluation matérielle,
        # mais le laisser à l'infini ferait exploser la cible d'entraînement.
        q[pv[0].uci()] = info["score"].relative.score(mate_score=10_000)
    if not q:
        return None
    return json.dumps({"prefixe": d["prefixe"], "q": q}, separators=(",", ":"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="entree",
                   default="data/q/candidats.jsonl")
    p.add_argument("--out",
                   default="data/q/q_labels.jsonl")
    p.add_argument("--stockfish",
                   default=os.environ.get("STOCKFISH", "stockfish"))
    p.add_argument("--workers", type=int, default=9)
    p.add_argument("--noeuds", type=int, default=200_000)
    p.add_argument("--limite", type=int, default=0,
                   help="0 = tout le fichier ; sinon n premières lignes")
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    print(f"[q] {args.workers} workers, {args.noeuds:,} noeuds par position",
          flush=True)

    t0 = time.time()
    n = ecrits = 0
    with open(args.entree) as f, open(args.out, "w") as g:
        lignes = f if not args.limite else (l for _, l in
                                            zip(range(args.limite), f))
        with mp.Pool(args.workers, initializer=init,
                     initargs=(args.stockfish, args.noeuds)) as pool:
            for res in pool.imap_unordered(traiter, lignes, chunksize=32):
                n += 1
                if res:
                    g.write(res + "\n")
                    ecrits += 1
                if n % 5000 == 0:
                    dt = time.time() - t0
                    g.flush()
                    print(f"  {n:,} lues / {ecrits:,} ecrites, "
                          f"{dt/60:.1f} min ({n/dt:.0f}/s)", flush=True)

    dt = time.time() - t0
    print(f"\n{ecrits:,} positions etiquetees -> {args.out}")
    print(f"duree : {dt/60:.1f} min ({n/max(dt,1):.0f} positions/s)")


if __name__ == "__main__":
    main()
