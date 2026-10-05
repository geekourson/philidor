"""Jeu de positions pour la mesure 3 : 20 000 milieux de partie de validation.

Pas de deduplication contre l'entrainement ici, et c'est deliberé : la mesure 3
teste ce que le modele REPRESENTE, pas s'il generalise. Le risque a ecarter est
que la SONDE surapprenne, et il l'est par le decoupage train/test de la sonde
elle-meme, sur des positions disjointes.

Positions tirees de la fin du fichier de validation, pour ne pas recouvrir les
5 000 du jeu principal qui viennent du debut.
"""
import argparse, json, chess, numpy as np

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--val", default="data/val_games.txt")
    p.add_argument("--n", type=int, default=20000)
    p.add_argument("--out", default="diag/sondes_positions.jsonl")
    p.add_argument("--seed", type=int, default=777)
    a = p.parse_args()

    rng = np.random.default_rng(a.seed)
    lignes = open(a.val).read().splitlines()
    lignes = lignes[len(lignes)//2:]          # seconde moitie du fichier
    out, vus = [], set()
    for l in lignes:
        c = l.split()
        if len(c) < 32 or len(out) >= a.n:
            continue
        n = int(rng.integers(30, min(len(c), 80)))
        b = chess.Board()
        try:
            for u in c[:n]: b.push(chess.Move.from_uci(u))
        except Exception:
            continue
        if b.is_game_over() or len(b.piece_map()) < 13:
            continue
        e = b.epd()
        if e in vus:
            continue
        vus.add(e)
        out.append({"prefixe": c[:n], "fen": b.fen()})
    with open(a.out, "w") as g:
        for d in out:
            g.write(json.dumps(d, separators=(",", ":")) + "\n")
    print(f"{len(out):,} positions -> {a.out}")


if __name__ == "__main__":
    main()
