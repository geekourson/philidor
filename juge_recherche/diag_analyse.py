"""Taux de gaffes et mesure 1 : proposition contre sélection.

Une gaffe = le coup argmax du modèle perd au moins 100 cp par rapport au
meilleur coup, les deux évalués par Stockfish à profondeur 18, du point de vue
du joueur au trait (donc aucune inversion pour les Noirs).

Mesure 1 : sur les positions gaffées, à quel rang le modèle place-t-il le
meilleur coup de Stockfish ? Les positions non gaffées servent de témoin. Un
écart marqué dirait que le bon coup est bien proposé mais mal choisi, un défaut
de sélection, que l'on pourrait corriger sans réentraîner.

Intervalles de Wilson à 95 %, adaptés aux proportions, y compris près de 0 et 1.

    python diag_analyse.py
"""

import argparse
import json
import math
import os


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 3
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def bloc(nom, rangs, n_tot):
    """Recapitulatif top-k et rang moyen pour un groupe de positions."""
    out = {"n": len(rangs)}
    for k in (1, 3, 5, 10):
        c = sum(1 for r in rangs if r is not None and r <= k)
        p, b, h = wilson(c, len(rangs))
        out[f"top{k}"] = {"n": c, "taux": round(p, 4),
                          "ic95": [round(b, 4), round(h, 4)]}
    connus = [r for r in rangs if r is not None]
    out["rang_moyen"] = round(sum(connus) / len(connus), 2) if connus else None
    out["rang_median"] = (sorted(connus)[len(connus) // 2] if connus else None)
    out["sans_rang"] = len(rangs) - len(connus)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--modele", default="diag/modele.jsonl")
    p.add_argument("--stockfish", default="diag/stockfish18.jsonl")
    p.add_argument("--seuil", type=int, default=100)
    p.add_argument("--out", default="diag/mesure1.json")
    p.add_argument("--gaffes", default="diag/gaffes.jsonl")
    args = p.parse_args()

    M = {json.loads(l)["fen"]: json.loads(l) for l in open(args.modele)}
    S = {json.loads(l)["fen"]: json.loads(l) for l in open(args.stockfish)}
    fens = [f for f in M if f in S]
    print(f"[analyse] {len(fens):,} positions appariees "
          f"(modele {len(M):,}, stockfish {len(S):,})")

    rangs_g, rangs_ng, gaffes = [], [], []
    pertes = []
    ecartes = 0
    for f in fens:
        m, s = M[f], S[f]
        cp = s["cp"]
        if not cp or len(cp) < s["n_legaux"]:
            ecartes += 1
            continue
        coup = m["classement"][0]
        if coup not in cp:
            ecartes += 1
            continue
        meilleur_cp = max(cp.values())
        perte = meilleur_cp - cp[coup]
        meilleur = max(cp, key=cp.get)
        rang = (m["classement"].index(meilleur) + 1
                if meilleur in m["classement"] else None)
        if perte >= args.seuil:
            rangs_g.append(rang)
            pertes.append(perte)
            gaffes.append({"fen": f, "coup_modele": coup,
                           "meilleur": meilleur, "perte": perte,
                           "rang_meilleur": rang})
        else:
            rangs_ng.append(rang)

    n = len(rangs_g) + len(rangs_ng)
    tp, tb, th = wilson(len(rangs_g), n)
    res = {
        "positions": n, "ecartees": ecartes, "seuil_cp": args.seuil,
        "taux_gaffes": {"n": len(rangs_g), "taux": round(tp, 4),
                        "ic95": [round(tb, 4), round(th, 4)]},
        "perte_moyenne_cp": round(sum(pertes) / len(pertes), 1) if pertes else None,
        "perte_mediane_cp": sorted(pertes)[len(pertes) // 2] if pertes else None,
        "gaffees": bloc("gaffees", rangs_g, n),
        "non_gaffees": bloc("non gaffees", rangs_ng, n),
    }
    rec = {}
    for k in (3, 5, 10):
        c = res["gaffees"][f"top{k}"]["n"]
        pr, br, hr = wilson(c, len(rangs_g))
        rec[f"k={k}"] = {"gaffes_recuperables": c,
                         "part_des_gaffes": round(pr, 4),
                         "ic95": [round(br, 4), round(hr, 4)],
                         "taux_gaffes_restant": round(
                             (len(rangs_g) - c) / n, 4)}
    res["selecteur_parfait"] = rec

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=2)
    with open(args.gaffes, "w") as g:
        for d in gaffes:
            g.write(json.dumps(d, separators=(",", ":")) + "\n")

    print(f"\ntaux de gaffes : {tp:.2%}  IC95 [{tb:.2%} ; {th:.2%}]  "
          f"({len(rangs_g):,}/{n:,})")
    print(f"perte moyenne  : {res['perte_moyenne_cp']} cp "
          f"(mediane {res['perte_mediane_cp']})")
    print(f"\n{'':<14}{'gaffees':>22}{'non gaffees':>22}")
    for k in (1, 3, 5, 10):
        a, b = res["gaffees"][f"top{k}"], res["non_gaffees"][f"top{k}"]
        print(f"  top-{k:<9}{a['taux']:>10.1%} [{a['ic95'][0]:.1%};{a['ic95'][1]:.1%}]"
              f"{b['taux']:>10.1%} [{b['ic95'][0]:.1%};{b['ic95'][1]:.1%}]")
    print(f"  rang moyen {res['gaffees']['rang_moyen']:>11}"
          f"{res['non_gaffees']['rang_moyen']:>22}")
    print(f"\nselecteur parfait :")
    for k, d in rec.items():
        print(f"  {k:<6} recupere {d['part_des_gaffes']:.1%} des gaffes "
              f"-> taux de gaffes {tp:.2%} -> {d['taux_gaffes_restant']:.2%}")
    print(f"\n-> {args.out} | {len(gaffes):,} gaffes -> {args.gaffes}")


if __name__ == "__main__":
    main()
