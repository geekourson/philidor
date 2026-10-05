"""Quelle mesure hors ligne suit l'Elo ? Pertes par sélecteur, sur les 5 000 positions.

Le taux de gaffes (perte >= 100 cp) a mal prédit le juge : 15,72 %, au niveau de
Stockfish profondeur 1 (14,94 %), mais +52 Elo en partie contre +127. Quatre
sélecteurs ont un Elo mesuré en duel contre l'argmax (argmax 0, juge +51,8,
Stockfish p1 +126,8, p4 +329,1). On calcule pour chacun plusieurs mesures de
perte, sur les mêmes positions, et on regarde laquelle les range dans le même
ordre et avec les mêmes écarts. Le réseau de valeur est placé sur cette échelle
AVANT son duel, pour que la prédiction soit écrite avant le résultat.
"""
import os
import json, math, multiprocessing as mp
import numpy as np
import torch
import chess, chess.engine
from model import ChessGPT, ModelConfig
from juge_recherche.juge_train import Juge
from juge_recherche.valeur_train import Valeur, charger_verrou, noter, taux

SF_BIN = os.environ.get("STOCKFISH", "stockfish")
SF = None; D = 1


def _init(d):
    global SF, D
    D = d
    SF = chess.engine.SimpleEngine.popen_uci(SF_BIN); SF.configure({"Threads": 1, "Hash": 16})


def _choisir(args):
    fen, top = args
    if len(top) == 1:
        return top[0]
    return SF.play(chess.Board(fen), chess.engine.Limit(depth=D),
                   root_moves=[chess.Move.from_uci(u) for u in top], game=object()).move.uci()


def main():
    dev = "cuda:0"; v = json.load(open("data/vocab.json"))
    D_ = charger_verrou("diag/positions.jsonl", "diag/modele.jsonl", "diag/stockfish18.jsonl",
                        v["stoi"], v["bos_id"], 256)
    P_ = charger_verrou("diag/propres_positions.jsonl", "diag/propres_modele.jsonl",
                        "diag/propres_sf18.jsonl", v["stoi"], v["bos_id"], 256)
    pos = {json.loads(l)["fen"] for l in open("diag/positions.jsonl")}
    sf = {json.loads(l)["fen"] for l in open("diag/stockfish18.jsonl")}
    fens = [json.loads(l)["fen"] for l in open("diag/modele.jsonl")]
    fens = [f for f in fens if f in pos and f in sf]
    assert len(fens) == len(D_)

    ck = torch.load("checkpoints/valeur.pt", map_location="cpu", weights_only=False)
    m = Valeur(ChessGPT(ModelConfig(**ck["model_config"]))); m.load_state_dict(ck["modele"])
    m = m.to(dev).eval(); noter(m, D_, v["pad_id"], dev); noter(m, P_, v["pad_id"], dev)
    ALS = (0.0, 0.1, 0.25, 0.5, 1.0, 2.0)
    reg = {al: taux(P_, 5, al) for al in ALS}; al_v = min(ALS, key=lambda a: reg[a][0] / reg[a][1])

    jk = torch.load("checkpoints/juge.pt", map_location="cpu", weights_only=False)
    J = Juge(jk["d"]); J.load_state_dict(jk["etat"]); J = J.to(dev).eval()
    X = np.load("data/juge/eval_diag.X.npy")
    meta = json.load(open("data/juge/eval_diag.meta.json"))
    with torch.no_grad():
        s = J(torch.tensor(X, dtype=torch.float32, device=dev)).cpu().numpy()
    M = [json.loads(l) for l in open("diag/modele.jsonl")]
    nj = {}
    for i, mm in enumerate(meta):
        nj.setdefault(M[mm["r"]]["fen"], {})[mm["u"]] = float(s[i])

    def choix(r, notes, al):
        c = [(u, n, lp) for u, n, lp in zip(r["cands"], notes, r["lp"]) if u in r["cp"]]
        return max(c, key=lambda t: t[1] + al * t[2])[0]

    C = {"argmax": [r["cands"][0] for r in D_],
         "juge": [choix(r, [nj[f].get(u, -1e9) for u in r["cands"]], 0.5) for r, f in zip(D_, fens)],
         f"valeur (alpha {al_v})": [choix(r, r["v"], al_v) for r in D_]}
    for d in (1, 4):
        with mp.Pool(10, initializer=_init, initargs=(d,)) as pool:
            C[f"stockfish p{d}"] = pool.map(_choisir, [(f, r["cands"]) for f, r in zip(fens, D_)], chunksize=50)

    elo = {"argmax": 0.0, "juge": 51.8, "stockfish p1": 126.8, "stockfish p4": 329.1}
    print(f"{'selecteur':<22}{'Elo duel':>9}{'>=100cp':>9}{'>=50cp':>9}{'>=200cp':>9}"
          f"{'perte moy.':>11}{'perte moy. <=300':>18}")
    res = {}
    for nom, ch in C.items():
        pertes = []
        for r, u in zip(D_, ch):
            if u not in r["cp"]:
                continue
            pertes.append(max(r["cp"].values()) - r["cp"][u])
        p = np.array(pertes, float)
        res[nom] = {"n": len(p), "g100": float((p >= 100).mean()), "g50": float((p >= 50).mean()),
                    "g200": float((p >= 200).mean()), "moy": float(np.minimum(p, 1000).mean()),
                    "moy300": float(np.minimum(p, 300).mean())}
        e = f"{elo[nom]:+.0f}" if nom in elo else "?"
        q = res[nom]
        print(f"{nom:<22}{e:>9}{q['g100']:>9.2%}{q['g50']:>9.2%}{q['g200']:>9.2%}"
              f"{q['moy']:>11.1f}{q['moy300']:>18.1f}")
    json.dump(res, open("diag/valeur_pertes.json", "w"), indent=2)


if __name__ == "__main__":
    main()
