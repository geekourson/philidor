"""Rendu d'une partie Lichess en GIF animé + diagramme de la position clé.

Sert à illustrer la première victoire du bot philidor-142M sur Lichess
(partie 0ofezMba, geekours 0-1 philidor-142M).

    python game_gif.py --pgn "chemin.pgn" --out-dir figures

Aucune dépendance exotique : python-chess pour la logique, matplotlib pour le
rendu (pièces en glyphes Unicode pleins, colorés blanc/noir avec un liseré de
contraste), PIL pour assembler le GIF. Pas de cairosvg ni d'imageio.
"""

import argparse
import io

import chess
import chess.pgn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patheffects
from PIL import Image

# Thème Lichess (vert), pour rester cohérent avec l'endroit où la partie a eu lieu.
CLAIR = "#ebecd0"
FONCE = "#779556"
SURLIGNE = "#f6f669"      # dernier coup
GLYPHE = {"k": "♚", "q": "♛", "r": "♜",
          "b": "♝", "n": "♞", "p": "♟"}


def rendu_position(board: chess.Board, dernier: chess.Move | None,
                   titre: str, sous_titre: str) -> Image.Image:
    """Rend une position en image PIL (échiquier vu côté Blancs)."""
    fig, ax = plt.subplots(figsize=(4.8, 5.3), dpi=110)
    ax.set_xlim(0, 8); ax.set_ylim(0, 8)
    ax.set_aspect("equal"); ax.axis("off")

    cases_surlignees = set()
    if dernier is not None:
        cases_surlignees = {dernier.from_square, dernier.to_square}

    for sq in chess.SQUARES:
        f, r = chess.square_file(sq), chess.square_rank(sq)
        base = CLAIR if (f + r) % 2 else FONCE
        ax.add_patch(plt.Rectangle((f, r), 1, 1, color=base))
        if sq in cases_surlignees:
            ax.add_patch(plt.Rectangle((f, r), 1, 1, color=SURLIGNE, alpha=0.55))
        piece = board.piece_at(sq)
        if piece:
            couleur = "white" if piece.color == chess.WHITE else "#111111"
            contour = "#111111" if piece.color == chess.WHITE else "#f4f4f4"
            ax.text(f + 0.5, r + 0.5, GLYPHE[piece.symbol().lower()],
                    fontsize=30, ha="center", va="center", color=couleur,
                    path_effects=[patheffects.withStroke(linewidth=2.2,
                                                         foreground=contour)])

    # coordonnées discrètes
    for f in range(8):
        ax.text(f + 0.5, -0.02, "abcdefgh"[f], ha="center", va="top",
                fontsize=8, color="#555")
    for r in range(8):
        ax.text(-0.02, r + 0.5, str(r + 1), ha="right", va="center",
                fontsize=8, color="#555")

    ax.set_title(titre, fontsize=14, fontweight="bold", pad=10, color="#222")
    fig.text(0.5, 0.045, sous_titre, ha="center", fontsize=10, color="#555")
    fig.subplots_adjust(left=0.04, right=0.98, top=0.9, bottom=0.09)

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pgn", required=True)
    p.add_argument("--out-dir", default="figures")
    p.add_argument("--gif", default="lichess_win.gif")
    p.add_argument("--cle", default="lichess_win_fork.png",
                   help="diagramme de la position clé")
    p.add_argument("--cle-ply", type=int, default=34,
                   help="demi-coup de la position clé (défaut 34 = 17...Nxg5)")
    p.add_argument("--ms-par-coup", type=int, default=850,
                   help="durée d'une image en ms (baisser pour les longues parties)")
    p.add_argument("--cle-titre", default="La position clé",
                   help="titre du diagramme de la position clé")
    args = p.parse_args()

    with open(args.pgn) as f:
        game = chess.pgn.read_game(f)
    blancs = game.headers.get("White", "?")
    noirs = game.headers.get("Black", "?")
    resultat = game.headers.get("Result", "*")
    ouverture = game.headers.get("Opening", "")

    board = game.board()
    frames = []
    # position initiale
    frames.append(rendu_position(
        board, None, f"{blancs}  vs  {noirs}",
        ouverture or "Position initiale"))

    moves = list(game.mainline_moves())
    cle_img = None
    for i, mv in enumerate(moves, start=1):
        san = board.san(mv)
        board.push(mv)
        num = (i + 1) // 2
        trait = "Blancs" if i % 2 == 1 else "Noirs"
        sous = f"{num}.{'' if i % 2 == 1 else '..'} {san}   ({trait})"
        img = rendu_position(board, mv, f"{blancs}  vs  {noirs}", sous)
        frames.append(img)
        if i == args.cle_ply:
            cle_img = rendu_position(board, mv, args.cle_titre, sous)

    # image finale : résultat
    frames.append(rendu_position(
        board, moves[-1] if moves else None,
        f"{blancs}  {resultat}  {noirs}",
        "philidor-142M gagne, 0 coup illégal"))

    out_gif = f"{args.out_dir}/{args.gif}"
    # tenir la 1re et la dernière frame plus longtemps
    durees = [1400] + [args.ms_par_coup] * (len(frames) - 2) + [4000]
    frames[0].save(out_gif, save_all=True, append_images=frames[1:],
                   duration=durees, loop=0, optimize=True)
    print(f"GIF   : {out_gif}  ({len(frames)} frames)")

    if cle_img is not None:
        out_cle = f"{args.out_dir}/{args.cle}"
        cle_img.save(out_cle)
        print(f"clé   : {out_cle}")


if __name__ == "__main__":
    main()
