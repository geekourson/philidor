"""Mesure 2 : à quelle profondeur la gaffe devient-elle visible ?

Pour chaque position gaffée, on cherche la profondeur Stockfish MINIMALE à
laquelle le coup du modèle apparaît perdant d'au moins 100 cp par rapport au
meilleur coup DE CETTE MÊME PROFONDEUR. Comparer le coup du modèle à
profondeur 3 avec le meilleur coup à profondeur 18 mélangerait deux questions.

Balayage de 1 à 12. Trois groupes :
  profondeur 1      : pièce laissée en prise, ou capture gratuite manquée ;
  profondeurs 2 à 4 : combinaison courte ;
  5 et plus         : calcul hors de portée d'une passe avant.

Convention de signe : `.relative`, du point de vue du joueur au trait, donc
aucune inversion pour les Noirs.

    python diag_profondeur.py --workers 10
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
PMAX = 12
SEUIL = 100


def _init(chemin, pmax, seuil):
    global SF, PMAX, SEUIL
    PMAX, SEUIL = pmax, seuil
    SF = chess.engine.SimpleEngine.popen_uci(chemin)
    SF.configure({"Threads": 1, "Hash": 64})


def _traiter(ligne):
    d = json.loads(ligne)
    b = chess.Board(d["fen"])
    coup = d["coup_modele"]
    n = b.legal_moves.count()
    par_prof = {}
    premiere = None
    for prof in range(1, PMAX + 1):
        try:
            infos = SF.analyse(b, chess.engine.Limit(depth=prof), multipv=n)
        except chess.engine.EngineError:
            return None
        cps = {}
        for info in infos:
            pv = info.get("pv")
            if pv:
                cps[pv[0].uci()] = info["score"].relative.score(mate_score=10_000)
        if coup not in cps or not cps:
            continue
        perte = max(cps.values()) - cps[coup]
        par_prof[prof] = perte
        if premiere is None and perte >= SEUIL:
            premiere = prof
    return json.dumps({"fen": d["fen"], "coup_modele": coup,
                       "profondeur_min": premiere, "pertes": par_prof},
                      separators=(",", ":"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gaffes", default="diag/gaffes.jsonl")
    p.add_argument("--out", default="diag/profondeur.jsonl")
    p.add_argument("--stockfish", default=SF_BIN)
    p.add_argument("--pmax", type=int, default=12)
    p.add_argument("--seuil", type=int, default=100)
    p.add_argument("--workers", type=int, default=10)
    args = p.parse_args()

    lignes = open(args.gaffes).read().splitlines()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    print(f"[prof] {len(lignes):,} gaffes, balayage 1..{args.pmax}", flush=True)
    t0 = time.time(); n = 0
    with open(args.out, "w") as g, mp.Pool(
            args.workers, initializer=_init,
            initargs=(args.stockfish, args.pmax, args.seuil)) as pool:
        for res in pool.imap_unordered(_traiter, lignes, chunksize=4):
            if res:
                g.write(res + "\n"); n += 1
            if n % 200 == 0:
                print(f"  {n:,}/{len(lignes):,} ({time.time()-t0:.0f}s)",
                      flush=True)
    print(f"\n{n:,} gaffes analysees -> {args.out} ({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
