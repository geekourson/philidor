"""Positions que la recherche évalue vraiment, pour entraîner le juge dessus.

Le juge a appris sur des parties humaines ; la recherche l'interroge surtout sur
des positions imaginées plusieurs coups à l'avance, que les humains atteignent
rarement. On fait jouer le moteur du bot contre lui-même (duel_lot.py, recherche
des deux côtés) en enregistrant un échantillon des feuilles évaluées
(recherche.COLLECTE) : chaque feuille est la partie entière jusqu'à elle.

Sortie : une ligne JSON par position distincte, {"uci": [...]}, à noter par
Stockfish (etiqueter_positions.py).
"""
import argparse, json, sys
from juge_recherche import recherche
from juge_recherche import duel_lot


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--games", type=int, default=1000)
    p.add_argument("--sims", type=int, default=64)
    p.add_argument("--taux", type=float, default=0.25, help="fraction des feuilles enregistrées")
    p.add_argument("--valeur", default="checkpoints/valeur.pt")
    p.add_argument("--seed", type=int, default=31)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    recherche.COLLECTE, recherche.TAUX_COLLECTE = [], a.taux
    spec = f"recherche:{a.sims}:0.1:0.0:5"
    sys.argv = ["duel_lot", "--a", spec, "--b", spec, "--valeur-a", a.valeur, "--valeur-b", a.valeur,
                "--games", str(a.games), "--seed", str(a.seed), "--label-a", "bot", "--label-b", "bot",
                "--out", a.out + ".duel.json"]
    duel_lot.main()
    vus, n = set(), 0
    with open(a.out, "w") as f:
        for uci in recherche.COLLECTE:
            cle = " ".join(uci)
            if cle in vus or len(uci) < 2:
                continue
            vus.add(cle); f.write(json.dumps({"uci": uci}) + "\n"); n += 1
    print(f"[feuilles] {len(recherche.COLLECTE):,} feuilles enregistrées, {n:,} positions distinctes -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
