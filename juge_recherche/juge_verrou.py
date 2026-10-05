"""Étape 1, le verrou hors ligne : le juge fait-il baisser le taux de gaffes ?

Sur les positions dont TOUS les coups sont évalués à profondeur 18 : le juge
choisit parmi les k premiers coups du modèle, et on juge ce choix avec la table.
Aucune partie jouée. Repères, mesurés sur les mêmes positions :
    argmax 18,32 % | Stockfish prof. 1 : 14,94 % | prof. 4 : 11,88 % | parfait 2,12 %
Il faut battre la profondeur 1 pour valoir quelque chose.

Variante : note du juge + alpha x log-probabilité de la politique. Alpha est
réglé sur UN jeu (les positions du bot) et publié sur l'AUTRE (le diagnostic),
pour ne pas régler le paramètre sur les données où on le mesure.
"""
import argparse, json, math, sys
import numpy as np
import torch
from juge_recherche.juge_train import Juge


def wilson(k, n, z=1.96):
    p = k/n; d = 1+z*z/n; c = (p+z*z/(2*n))/d
    h = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return p, max(0, c-h), min(1, c+h)


def evaluer(prefixe, modele, sf, juge, dev):
    X = np.load(prefixe + ".X.npy"); meta = json.load(open(prefixe + ".meta.json"))
    with torch.no_grad():
        s = juge(torch.tensor(X, dtype=torch.float32, device=dev)).cpu().numpy()
    M = [json.loads(l) for l in open(modele)]
    S = {json.loads(l)["fen"]: json.loads(l)["cp"] for l in open(sf)}
    notes = {}
    for i, m in enumerate(meta):
        notes.setdefault(m["r"], {})[m["u"]] = float(s[i])
    lignes = []
    for r, nt in notes.items():
        fen = M[r]["fen"]
        if fen not in S: continue
        cls, pp = M[r]["classement"], M[r]["p"]
        lignes.append((S[fen], cls, pp, nt))
    return lignes


def taux(lignes, k, alpha):
    g = n = 0
    for cp, cls, pp, nt in lignes:
        cands = [u for u in cls[:k] if u in nt and u in cp]
        if not cands: continue
        lp = {u: math.log(max(pp[cls.index(u)], 1e-9)) for u in cands}
        choix = max(cands, key=lambda u: nt[u] + alpha * lp[u])
        n += 1; g += (max(cp.values()) - cp[choix]) >= 100
    return g, n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--juge", default="checkpoints/juge.pt")
    p.add_argument("--device", default="cuda:0")
    a = p.parse_args()
    ck = torch.load(a.juge, map_location="cpu", weights_only=False)
    juge = Juge(ck["d"]); juge.load_state_dict(ck["etat"]); juge = juge.to(a.device).eval()
    J = "data/juge/"
    diag = evaluer(J + "eval_diag", "diag/modele.jsonl", "diag/stockfish18.jsonl", juge, a.device)
    prop = evaluer(J + "eval_propres", "diag/propres_modele.jsonl", "diag/propres_sf18.jsonl", juge, a.device)

    print("=== juge seul (alpha = 0), sans aucun reglage ===")
    print(f"{'':<14}{'diagnostic':>28}{'positions du bot':>30}")
    for k in (1, 2, 3, 5):
        r = []
        for L in (diag, prop):
            g, n = taux(L, k, 0.0); pr = wilson(g, n)
            r.append(f"{pr[0]:>8.2%} [{pr[1]:.2%} ; {pr[2]:.2%}]")
        lab = "argmax (k=1)" if k == 1 else f"juge, top-{k}"
        print(f"{lab:<14}{r[0]:>28}{r[1]:>30}")

    print("\n=== juge + alpha x log p(politique), alpha regle sur les positions du bot ===")
    res = {}
    for alpha in (0.0, 0.25, 0.5, 1.0, 2.0, 4.0):
        g, n = taux(prop, 5, alpha); res[alpha] = g / n
        print(f"  alpha {alpha:<5} positions du bot (reglage) : {g/n:.2%}")
    best = min(res, key=res.get)
    g, n = taux(diag, 5, best); pr = wilson(g, n)
    print(f"\nalpha retenu = {best} -> DIAGNOSTIC (jamais vu par le reglage) : "
          f"{pr[0]:.2%} [{pr[1]:.2%} ; {pr[2]:.2%}]")
    print("reperes diagnostic : argmax 18,32 % | Stockfish p1 14,94 % | p4 11,88 % | p12 5,14 % | parfait 2,12 %")


if __name__ == "__main__":
    main()
