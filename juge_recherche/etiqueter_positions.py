"""Stockfish note des positions pour le juge : parties du bot et feuilles de la recherche.

Entrée, une ligne JSON par partie ou position : {"uci": [...]} (feuilles,
generer_feuilles.py), ou les PGN écrits par lichess-bot (--pgn-dir).
Sortie au format des parties annotées de Lichess, que valeur_encoder.py lit
directement : {"uci": [...], "cp": [...]}, cp[i] = évaluation APRÈS le coup i,
du point de vue des BLANCS, mats bornés à ±10000, None si non notée.

Deux modes :
  --mode feuille : seule la dernière position est notée (une cible par séquence) ;
  --mode partie  : toutes les positions de la partie sont notées.

Fins de partie notées exactement (mat ±10000, pat et matériel insuffisant 0),
sans appeler Stockfish. Processus en nice 19 : le bot passe devant.
"""
import argparse, glob, json, os, time
import multiprocessing as mp
import chess, chess.engine, chess.pgn

SF = None
LIM = None


def _init(chemin, noeuds):
    global SF, LIM
    os.nice(19)
    SF = chess.engine.SimpleEngine.popen_uci(chemin)
    SF.configure({"Threads": 1, "Hash": 32})
    LIM = chess.engine.Limit(nodes=noeuds)


def note(b):
    if b.is_checkmate():
        return -10000 if b.turn == chess.WHITE else 10000
    if b.is_stalemate() or b.is_insufficient_material():
        return 0
    s = SF.analyse(b, LIM)["score"].white().score(mate_score=10000)
    return max(-10000, min(10000, s))


def _traiter(args):
    uci, mode = args
    b = chess.Board(); cp = []
    try:
        for i, u in enumerate(uci):
            b.push_uci(u)
            cp.append(note(b) if (mode == "partie" or i == len(uci) - 1) else None)
    except Exception:
        return None
    return json.dumps({"uci": uci, "cp": cp}, separators=(",", ":"))


def lire_pgn(dossier, bot):
    for f in sorted(glob.glob(os.path.join(dossier, "*.pgn"))):
        with open(f) as h:
            g = chess.pgn.read_game(h)
        if g is None or g.headers.get("Variant", "Standard") not in ("Standard", "") or "FEN" in g.headers:
            continue
        if bot not in (g.headers.get("White"), g.headers.get("Black")):
            continue
        uci = [m.uci() for m in g.mainline_moves()]
        if len(uci) >= 10:
            yield uci


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="entree", default="")
    p.add_argument("--pgn-dir", default="")
    p.add_argument("--bot", default="philidor-142M")
    p.add_argument("--mode", choices=["feuille", "partie"], required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--stockfish", default=os.environ.get("STOCKFISH", "stockfish"))
    p.add_argument("--noeuds", type=int, default=100_000)
    p.add_argument("--workers", type=int, default=9)
    p.add_argument("--max", type=int, default=0)
    a = p.parse_args()
    if a.pgn_dir:
        source = list(lire_pgn(a.pgn_dir, a.bot))
    else:
        source = [json.loads(l)["uci"] for l in open(a.entree)]
    if a.max:
        source = source[:a.max]
    n_pos = sum(len(u) for u in source) if a.mode == "partie" else len(source)
    print(f"[etiquetage] {len(source):,} sequences, {n_pos:,} positions a noter, Stockfish {a.noeuds:,} noeuds, "
          f"{a.workers} processus", flush=True)
    t0 = time.time(); n = 0
    with open(a.out, "w") as g, mp.Pool(a.workers, initializer=_init, initargs=(a.stockfish, a.noeuds)) as pool:
        for r in pool.imap_unordered(_traiter, ((u, a.mode) for u in source), chunksize=4):
            if r:
                g.write(r + "\n"); n += 1
            if n % 2000 == 0 and n:
                g.flush()
                print(f"  {n:,}/{len(source):,} ({(time.time()-t0)/60:.1f} min)", flush=True)
    print(f"[etiquetage] {n:,} sequences notees -> {a.out} | {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
