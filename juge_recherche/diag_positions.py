"""Jeu de positions du diagnostic : 5 000 milieux de partie jamais vus.

Critères : demi-coups 30 à 80 (coups 15 à 40, donc hors de tout livre), plus de
12 pièces sur l'échiquier (frontière « finale » du reste du projet), partie non
terminée, tirées du jeu de validation.

DÉDUPLICATION PAR FEN, ET LE PIÈGE QU'ELLE CACHE. Une première version hachait
les positions d'un échantillon de `games_uci.txt` puis écartait les candidats
qui y figuraient : 84,65 % de collision, un chiffre absurde pour des milieux de
partie. La cause : **`val_games.txt` est un sous-ensemble de `games_uci.txt`**.
Le découpage train/validation se fait plus tard, par un masque `is_val`, et le
fichier texte des parties contient tout. On haché donc les parties de validation
elles-mêmes, et on mesurait une auto-collision.

Le sens est inversé ici, ce qui est à la fois exact et bien moins coûteux :
  1. on extrait les candidats de validation et on hache leurs EPD ;
  2. on parcourt TOUT le corpus d'entraînement, en sautant à l'identique les
     lignes qui sont des parties de validation, et on marque tout candidat dont
     l'EPD y apparaît ;
  3. on garde les premiers non contaminés.

L'EPD est le FEN sans les compteurs de coups : pièces, trait, roques, prise en
passant. C'est l'identité d'une position.

    python diag_positions.py --n 5000
"""

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import time

import chess

PLY_MIN, PLY_MAX = 30, 80
PIECES_MIN = 13
CIBLES = set()


def cle(board):
    return hashlib.blake2b(board.epd().encode(), digest_size=8).digest()


def _init(cibles):
    global CIBLES
    CIBLES = cibles


def _contaminants(ligne):
    """Cles de milieu de partie de cette partie d'entrainement qui visent un
    candidat. On ne renvoie que les collisions, donc presque toujours rien."""
    c = ligne.split()
    if len(c) < PLY_MIN + 2:
        return ()
    b = chess.Board()
    trouve = []
    try:
        for i, u in enumerate(c):
            if i >= PLY_MAX:
                break
            if i >= PLY_MIN and len(b.piece_map()) >= PIECES_MIN:
                k = cle(b)
                if k in CIBLES:
                    trouve.append(k)
            b.push(chess.Move.from_uci(u))
    except (ValueError, AssertionError):
        return ()
    return tuple(trouve)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--val", default="data/val_games.txt")
    p.add_argument("--train", default="data/games_uci.txt")
    p.add_argument("--n", type=int, default=5000)
    p.add_argument("--n-cands", type=int, default=40000)
    p.add_argument("--workers", type=int, default=10)
    p.add_argument("--out", default="diag/positions.jsonl")
    p.add_argument("--seed", type=int, default=20260922)
    args = p.parse_args()

    t0 = time.time()
    import numpy as np
    rng = np.random.default_rng(args.seed)

    print("[1/3] extraction des candidats de validation", flush=True)
    parties_val = set()
    cands, vus = [], {}
    for ligne in open(args.val):
        parties_val.add(ligne.strip())
        c = ligne.split()
        if len(c) < PLY_MIN + 2 or len(cands) >= args.n_cands:
            continue
        n = int(rng.integers(PLY_MIN, min(len(c), PLY_MAX)))
        b = chess.Board()
        try:
            for u in c[:n]:
                b.push(chess.Move.from_uci(u))
        except (ValueError, AssertionError):
            continue
        if b.is_game_over() or len(b.piece_map()) < PIECES_MIN:
            continue
        k = cle(b)
        if k in vus:
            continue
        vus[k] = len(cands)
        cands.append({"prefixe": c[:n], "fen": b.fen(), "ply": n,
                      "trait": "b" if b.turn else "n",
                      "n_legaux": b.legal_moves.count()})
    print(f"  {len(parties_val):,} parties de validation", flush=True)
    print(f"  {len(cands):,} candidats distincts", flush=True)

    print(f"[2/3] balayage du corpus d'entrainement complet", flush=True)
    cibles = set(vus)
    contamines = set()
    n_train = n_sautees = 0
    with open(args.train) as f, mp.Pool(args.workers, initializer=_init,
                                        initargs=(cibles,)) as pool:
        def flux():
            nonlocal n_sautees
            for ligne in f:
                s = ligne.strip()
                if s in parties_val:      # c'est une partie de validation
                    n_sautees += 1
                    continue
                yield s
        for r in pool.imap_unordered(_contaminants, flux(), chunksize=800):
            n_train += 1
            contamines.update(r)
            if n_train % 1_000_000 == 0:
                print(f"  {n_train:,} parties, {len(contamines):,} candidats "
                      f"contamines ({time.time()-t0:.0f}s)", flush=True)
    print(f"  {n_train:,} parties d'entrainement balayees", flush=True)
    print(f"  {n_sautees:,} parties de validation sautees", flush=True)
    print(f"  candidats vus a l'entrainement : {len(contamines):,} "
          f"({len(contamines)/len(cands):.2%})", flush=True)

    print("[3/3] ecriture", flush=True)
    gardes = [d for k, d in ((cle(chess.Board(c["fen"])), c) for c in cands)
              if k not in contamines][:args.n]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as g:
        for d in gardes:
            g.write(json.dumps(d, separators=(",", ":")) + "\n")
    blancs = sum(1 for d in gardes if d["trait"] == "b")
    print(f"\n{len(gardes):,} positions -> {args.out}")
    print(f"  trait aux Blancs : {blancs:,} ({blancs/len(gardes):.1%})")
    print(f"  coups legaux     : moyenne "
          f"{sum(d['n_legaux'] for d in gardes)/len(gardes):.1f}")
    print(f"  demi-coup        : moyenne "
          f"{sum(d['ply'] for d in gardes)/len(gardes):.1f}")
    print(f"duree : {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
