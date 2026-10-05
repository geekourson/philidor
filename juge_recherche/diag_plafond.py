"""Étape 0, hors ligne : taux de gaffes d'un sélecteur Stockfish sur le top-k.

Sur les 5 000 positions du diagnostic, dont TOUS les coups sont déjà évalués à
profondeur 18 : Stockfish à profondeur d choisit parmi le top-k de Philidor, et
on juge ce choix avec la table de profondeur 18. Ça dit, avant de jouer une
seule partie, quelle profondeur approche le plafond de 5,22 % trouvé par la
mesure 1 (sélecteur parfait = profondeur 18 elle-même).
"""
import os
import argparse, json, math, multiprocessing as mp, time
import chess, chess.engine

SF_BIN = os.environ.get("STOCKFISH", "stockfish")
SF = None; D = 4; K = 5

def _init(d, k):
    global SF, D, K
    D, K = d, k
    SF = chess.engine.SimpleEngine.popen_uci(SF_BIN)
    SF.configure({"Threads": 1, "Hash": 16})

def _choisir(args):
    fen, top = args
    b = chess.Board(fen)
    top = top[:K]
    if len(top) == 1:
        return fen, top[0]
    r = SF.play(b, chess.engine.Limit(depth=D),
                root_moves=[chess.Move.from_uci(u) for u in top], game=object())
    return fen, r.move.uci()

def wilson(k, n, z=1.96):
    p = k/n; d = 1+z*z/n; c = (p+z*z/(2*n))/d
    h = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return p, max(0, c-h), min(1, c+h)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--profondeurs", default="1,2,4,8,12")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--workers", type=int, default=10)
    p.add_argument("--out", default="diag/plafond.json")
    a = p.parse_args()

    S = {json.loads(l)["fen"]: json.loads(l) for l in open("diag/stockfish18.jsonl")}
    M = {json.loads(l)["fen"]: json.loads(l) for l in open("diag/modele.jsonl")}
    fens = [f for f in M if f in S]

    def taux(choix):
        k = n = 0
        for f, u in choix.items():
            cp = S[f]["cp"]
            if u not in cp: continue
            n += 1; k += (max(cp.values()) - cp[u]) >= 100
        return k, n

    res = {}
    k0, n0 = taux({f: M[f]["classement"][0] for f in fens})
    p0 = wilson(k0, n0); res["argmax"] = p0
    print(f"{'selecteur':<22}{'gaffes':>9}   IC 95 %")
    print(f"{'argmax (reference)':<22}{p0[0]:>9.2%}   [{p0[1]:.2%} ; {p0[2]:.2%}]", flush=True)
    kp, np_ = taux({f: max(M[f]["classement"][:a.k], key=lambda u: S[f]["cp"].get(u, -1e9))
                    for f in fens})
    pp = wilson(kp, np_); res["parfait"] = pp
    print(f"{'parfait (prof. 18)':<22}{pp[0]:>9.2%}   [{pp[1]:.2%} ; {pp[2]:.2%}]", flush=True)
    for d in [int(x) for x in a.profondeurs.split(",")]:
        t0 = time.time()
        with mp.Pool(a.workers, initializer=_init, initargs=(d, a.k)) as pool:
            choix = dict(pool.imap_unordered(_choisir,
                         [(f, M[f]["classement"]) for f in fens], chunksize=16))
        kk, nn = taux(choix); pr = wilson(kk, nn); res[f"prof{d}"] = pr
        print(f"{'Stockfish prof. '+str(d):<22}{pr[0]:>9.2%}   [{pr[1]:.2%} ; {pr[2]:.2%}]"
              f"   ({time.time()-t0:.0f}s)", flush=True)
    json.dump(res, open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
