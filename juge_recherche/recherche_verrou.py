"""Étape A, le verrou hors ligne : la recherche fait-elle baisser les pertes ?

Mêmes positions et même table de profondeur 18 que le diagnostic : la recherche
choisit un coup, la table le juge. Réglages (cpuct, fpu) choisis sur les
positions du bot, publiés sur le diagnostic, jamais l'inverse.

Contrôle intégré (--controle) : avec k = 5, 6 simulations et fpu = -10, chaque
coup du top-5 est visité exactement une fois ; le plus visité est alors départagé
par sa valeur, et la recherche doit redonner EXACTEMENT le choix « valeur seule »
du réseau de valeur. Tout écart est un bug de signe ou de chemin.

Repères diagnostic (sélection dans le top-5, étape 0 et valeur_pertes.py) :
    argmax 18,32 % | valeur 15,28 % | Stockfish p1 14,94 % | p2 13,18 % |
    p4 11,88 % | p8 7,46 % | p12 5,14 %
"""
import argparse, json, math, time
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
import torch
import chess
from model import ChessGPT, ModelConfig
from juge_recherche.valeur_train import Valeur, charger_verrou, noter
from juge_recherche.recherche import Recherche, chercher


def paris():
    return datetime.now(ZoneInfo("Europe/Paris")).strftime("%H:%M")


def positions(pos, mod, sf, stoi, bos):
    P = {json.loads(l)["fen"]: json.loads(l)["prefixe"] for l in open(pos)}
    S = {json.loads(l)["fen"]: json.loads(l)["cp"] for l in open(sf)}
    out = []
    for l in open(mod):
        m = json.loads(l)
        if m["fen"] not in P or m["fen"] not in S:
            continue
        b = chess.Board()
        for u in P[m["fen"]]:
            b.push_uci(u)
        assert b.board_fen() == chess.Board(m["fen"]).board_fen()
        out.append({"b": b, "ids": [bos] + [stoi[u] for u in P[m["fen"]]], "cp": S[m["fen"]],
                    "top": m["classement"][:5]})
    return out


def mesures(X, choix):
    p = np.array([max(x["cp"].values()) - x["cp"][u] for x, u in zip(X, choix) if u in x["cp"]], float)
    return {"n": len(p), "g100": float((p >= 100).mean()), "g50": float((p >= 50).mean()),
            "g200": float((p >= 200).mean()), "moy": float(np.minimum(p, 1000).mean()),
            "moy300": float(np.minimum(p, 300).mean())}


def wilson(p, n, z=1.96):
    d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0, c - h), min(1, c + h)


def lancer(X, pol, val, stoi, pad, dev, sims, cpuct, fpu, k, lot, par):
    choix = []
    for i in range(0, len(X), par):
        R = [Recherche(x["b"], x["ids"], stoi, k=k, cpuct=cpuct, fpu=fpu) for x in X[i:i + par]]
        chercher(R, pol, val, sims, pad, dev, lot=lot)
        choix += [r.choix().uci() for r in R]
    return choix


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--jeu", choices=["diag", "propres"], required=True)
    p.add_argument("--n", type=int, default=0, help="premières n positions seulement")
    p.add_argument("--configs", required=True, help="sims:cpuct:fpu:k,... ")
    p.add_argument("--valeur", default="checkpoints/valeur.pt")
    p.add_argument("--lot", type=int, default=8)
    p.add_argument("--par", type=int, default=256)
    p.add_argument("--controle", action="store_true")
    p.add_argument("--out", default="")
    p.add_argument("--device", default="cuda:0")
    a = p.parse_args()
    dev = a.device

    v = json.load(open("data/vocab.json")); stoi, bos, pad = v["stoi"], v["bos_id"], v["pad_id"]
    ck = torch.load("checkpoints/run2_best.pt", map_location="cpu", weights_only=False)
    pol = ChessGPT(ModelConfig(**ck["model_config"])); pol.load_state_dict(ck["model"]); pol = pol.to(dev).eval()
    cv = torch.load(a.valeur, map_location="cpu", weights_only=False)
    val = Valeur(ChessGPT(ModelConfig(**cv["model_config"]))); val.load_state_dict(cv["modele"])
    val = val.to(dev).eval()

    f = {"diag": ("diag/positions.jsonl", "diag/modele.jsonl", "diag/stockfish18.jsonl"),
         "propres": ("diag/propres_positions.jsonl", "diag/propres_modele.jsonl", "diag/propres_sf18.jsonl")}[a.jeu]
    X = positions(*f, stoi, bos)
    if a.n:
        X = X[:a.n]
    R = charger_verrou(*f, stoi, bos, 256)
    if a.n:
        R = R[:a.n]
    noter(val, R, pad, dev)
    seule = [r["cands"][int(np.argmax(r["v"]))] for r in R]
    base = {"argmax": mesures(X, [x["top"][0] for x in X]), "valeur seule top-5": mesures(X, seule)}
    print(f"[{paris()}] {a.jeu} : {len(X)} positions", flush=True)
    for nom, m in base.items():
        print(f"  {nom:<34} >=100 {m['g100']:.2%} | >=50 {m['g50']:.2%} | moy {m['moy']:.1f} | moy300 {m['moy300']:.1f}", flush=True)

    if a.controle:
        t0 = time.time()
        c = lancer(X, pol, val, stoi, pad, dev, 6, 1.0, -10.0, 5, a.lot, a.par)
        acc = sum(x == y for x, y in zip(c, seule)) / len(c)
        print(f"[controle] recherche 6 sim., fpu -10 == valeur seule : {acc:.2%} ({time.time()-t0:.0f}s)", flush=True)

    res = []
    for cfg in a.configs.split(","):
        sims, cpuct, fpu, k = cfg.split(":"); sims, k = int(sims), int(k); cpuct, fpu = float(cpuct), float(fpu)
        t0 = time.time()
        c = lancer(X, pol, val, stoi, pad, dev, sims, cpuct, fpu, k, a.lot, a.par)
        m = mesures(X, c); lo, hi = wilson(m["g100"], m["n"])
        m.update({"sims": sims, "cpuct": cpuct, "fpu": fpu, "k": k, "jeu": a.jeu,
                  "bas": lo, "haut": hi, "s_par_position": (time.time() - t0) / len(X),
                  "accord_valeur_seule": sum(x == y for x, y in zip(c, seule)) / len(c),
                  "choix": c, "gaffe": [int(u in x["cp"] and max(x["cp"].values()) - x["cp"][u] >= 100)
                                        for x, u in zip(X, c)]})
        res.append(m)
        print(f"[{paris()}] sims {sims:>4} cpuct {cpuct:<4} fpu {fpu:<5} k {k} : >=100 {m['g100']:.2%} "
              f"[{lo:.2%} ; {hi:.2%}] | >=50 {m['g50']:.2%} | moy {m['moy']:.1f} | moy300 {m['moy300']:.1f} "
              f"| {m['s_par_position']*1000:.0f} ms/pos", flush=True)
        if a.out:
            json.dump({"base": base, "res": res}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
