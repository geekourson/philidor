"""Le visuel du post 09 : deux cartes graphiques, deux coups différents.

Format carré (1200x1200), pensé pour le fil mobile LinkedIn.

Données réelles : un des 2 désaccords mesurés sur 500 positions inédites
(logs/thinking/desaccord_exemple.json). Le prefixe rejoue la partie jusqu'à la
position litigieuse, on rend le meme echiquier deux fois avec la fleche que
chaque carte a choisie, et on pose dessous les notes qui expliquent l'ecart.

Rendu maison en matplotlib + glyphes Unicode, comme game_gif.py : pas de
cairosvg, et la meme identite visuelle que les autres figures du depot.
"""

import argparse
import json
import os

import chess
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patheffects

from plots import INK_MUTED, INK_PRIMARY, INK_SECONDARY, SERIES

CLAIR = "#ebecd0"
FONCE = "#779556"
GLYPHE = {"k": "♚", "q": "♛", "r": "♜",
          "b": "♝", "n": "♞", "p": "♟"}
PAS = 0.0625


def echiquier(ax, board, coup, couleur):
    """Rend la position vue cote Blancs, avec la fleche du coup choisi."""
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 8)
    ax.set_aspect("equal")
    ax.axis("off")

    for sq in chess.SQUARES:
        f, r = chess.square_file(sq), chess.square_rank(sq)
        ax.add_patch(plt.Rectangle((f, r), 1, 1,
                                   color=CLAIR if (f + r) % 2 else FONCE))
        if sq in (coup.from_square, coup.to_square):
            ax.add_patch(plt.Rectangle((f, r), 1, 1, color=couleur, alpha=0.28))
        piece = board.piece_at(sq)
        if piece:
            blanc = piece.color == chess.WHITE
            ax.text(f + 0.5, r + 0.47, GLYPHE[piece.symbol().lower()],
                    fontsize=26, ha="center", va="center",
                    color="white" if blanc else "#141414",
                    path_effects=[patheffects.withStroke(
                        linewidth=2.0,
                        foreground="#141414" if blanc else "#f2f2f2")],
                    zorder=3)

    x0 = chess.square_file(coup.from_square) + 0.5
    y0 = chess.square_rank(coup.from_square) + 0.5
    x1 = chess.square_file(coup.to_square) + 0.5
    y1 = chess.square_rank(coup.to_square) + 0.5
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=couleur, linewidth=6.5,
                                mutation_scale=34, alpha=0.92,
                                shrinkA=13, shrinkB=13,
                                path_effects=[patheffects.withStroke(
                                    linewidth=9.5, foreground="white")]),
                zorder=4)

    for f in range(8):
        ax.text(f + 0.5, -0.06, "abcdefgh"[f], ha="center", va="top",
                fontsize=8.5, color=INK_MUTED)
    for r in range(8):
        ax.text(-0.06, r + 0.5, str(r + 1), ha="right", va="center",
                fontsize=8.5, color=INK_MUTED)
    ax.add_patch(plt.Rectangle((0, 0), 8, 8, fill=False,
                               edgecolor="#c9c8c3", linewidth=1.2))


def fmt(x):
    return f"{x:g}".replace(".", ",")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--desaccord", default="results/desaccord_exemple.json")
    p.add_argument("--out", default="figures/determinisme_gpu.png")
    args = p.parse_args()
    d = json.load(open(args.desaccord))
    A, B = d["A"], d["B"]
    board = chess.Board()
    for u in d["prefixe"]:
        board.push_uci(u)

    cartes = []
    for nom, src, couleur in (("Carte A", A, SERIES[0]),
                              ("Carte B", B, SERIES[1])):
        uci = max(src, key=src.get) if len(set(src.values())) == len(src) \
            else list(src)[0]
        cartes.append((nom, src, couleur, chess.Move.from_uci(uci), uci))

    fig = plt.figure(figsize=(8, 8), dpi=150)

    fig.text(0.5, 0.965, "Mêmes poids. Aucun hasard.", ha="center", va="top",
             fontsize=27, fontweight="bold", color=INK_PRIMARY)
    fig.text(0.5, 0.913, "Et deux coups différents selon la carte graphique.",
             ha="center", va="top", fontsize=16, color=INK_SECONDARY)

    for i, (nom, src, couleur, coup, uci) in enumerate(cartes):
        gauche = 0.055 + i * 0.475
        fig.text(gauche + 0.2125, 0.862, nom, ha="center", va="top",
                 fontsize=15, fontweight="bold", color=INK_SECONDARY)
        ax = fig.add_axes([gauche, 0.395, 0.425, 0.425])
        echiquier(ax, board, coup, couleur)
        fig.text(gauche + 0.2125, 0.372, board.san(coup), ha="center", va="top",
                 fontsize=23, fontweight="bold", color=couleur)

    tab = fig.add_axes([0.055, 0.055, 0.89, 0.245])
    tab.set_xlim(0, 1)
    tab.set_ylim(0, 1)
    tab.axis("off")
    tab.add_patch(plt.Rectangle((0, 0), 1, 1, facecolor="#f2f1ed",
                                edgecolor="#e2e1dc", linewidth=1.2))

    cx = {"bar": 0.028, "lab": 0.062, "m1": 0.36, "m2": 0.53, "note": 0.655}
    m1, m2 = "f4f7", "g3e4"
    for cle, mv in ((cx["m1"], m1), (cx["m2"], m2)):
        tab.text(cle, 0.855, board.san(chess.Move.from_uci(mv)), ha="center",
                 va="center", fontsize=14, fontweight="bold",
                 color=INK_SECONDARY)
    tab.text(cx["lab"], 0.855, "notes du modèle", ha="left", va="center",
             fontsize=12.5, color=INK_MUTED)
    tab.plot([0.028, 0.972], [0.715, 0.715], color="#dcdbd6", linewidth=1.1)

    for i, (nom, src, couleur, coup, uci) in enumerate(cartes):
        y = 0.545 - i * 0.205
        tab.add_patch(plt.Rectangle((cx["bar"], y - 0.055), 0.016, 0.11,
                                    color=couleur))
        tab.text(cx["lab"], y, nom, ha="left", va="center", fontsize=14,
                 color=INK_PRIMARY, fontweight="bold")
        for cle, mv in ((cx["m1"], m1), (cx["m2"], m2)):
            gagne = uci == mv
            tab.text(cle, y, fmt(src[mv]), ha="center", va="center",
                     fontsize=17, color=INK_PRIMARY if gagne else INK_MUTED,
                     fontweight="bold" if gagne else "normal")
        egal = src[m1] == src[m2]
        tab.text(cx["note"], y,
                 "exactement la même note" if egal else "un cran d'écart",
                 ha="left", va="center", fontsize=12, color=INK_SECONDARY)

    tab.text(0.5, 0.135,
             "Un cran vaut 0,0625 : la finesse de calcul d'une carte graphique.",
             ha="center", va="center", fontsize=12.5, color=INK_MUTED)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, facecolor="white")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
