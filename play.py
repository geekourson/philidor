"""Jouer contre le modèle, dans le terminal.

    python play.py                                  # partie contre run1, tu joues blanc
    python play.py --ckpt checkpoints/run2_best.pt  # contre run2
    python play.py --noir                           # tu joues noir
    python play.py --temperature 0.4                # un adversaire moins prévisible

Les coups s'entrent indifféremment en notation courante (`Cf3`, `e4`, `O-O`) ou
en UCI (`g1f3`, `e2e4`, `e1g1`). Commandes disponibles pendant la partie :

    ?          la liste des coups légaux
    top        ce que le modèle pense de la position actuelle
    annuler    revenir en arrière d'un coup complet
    abandon    terminer la partie
"""

import argparse
import os

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import chess

from engine import ChessEngine

PIECES = {
    "P": "♙", "N": "♘", "B": "♗", "R": "♖", "Q": "♕", "K": "♔",
    "p": "♟", "n": "♞", "b": "♝", "r": "♜", "q": "♛", "k": "♚",
}


def afficher(board: chess.Board, vue_blanc: bool = True):
    """Dessine l'échiquier, orienté du côté du joueur humain."""
    rangs = range(7, -1, -1) if vue_blanc else range(8)
    cols = range(8) if vue_blanc else range(7, -1, -1)
    print()
    for r in rangs:
        ligne = f"  {r + 1} "
        for c in cols:
            p = board.piece_at(chess.square(c, r))
            ligne += (PIECES[p.symbol()] if p else "·") + " "
        print(ligne)
    lettres = "abcdefgh" if vue_blanc else "hgfedcba"
    print("    " + " ".join(lettres))
    if board.is_check():
        print("\n  ÉCHEC")
    print()


def afficher_analyse(eng: ChessEngine, board: chess.Board, n=6):
    """Montre les coups que le modèle juge les plus probables.

    C'est la fenêtre la plus directe sur ce que le modèle a appris : les
    probabilités affichées sont celles qu'il attribue *après* masquage des
    coups illégaux, donc renormalisées sur les seuls coups jouables.
    """
    scores = eng.move_scores(board)
    top = sorted(scores.items(), key=lambda kv: -kv[1])[:n]
    print("  ce que le modèle envisage :")
    for uci, p in top:
        san = board.san(chess.Move.from_uci(uci))
        barre = "█" * max(1, round(p * 30))
        print(f"    {san:<8} {uci}  {p:6.1%} {barre}")
    print()


def lire_coup(board: chess.Board) -> chess.Move | str:
    """Lit un coup au clavier, en notation courante ou en UCI."""
    while True:
        txt = input("  ton coup > ").strip()
        if not txt:
            continue
        bas = txt.lower()
        if bas in ("?", "aide", "coups"):
            print("  " + " ".join(sorted(board.san(m) for m in board.legal_moves)))
            continue
        if bas in ("top", "analyse"):
            return "analyse"
        if bas in ("annuler", "undo"):
            return "annuler"
        if bas in ("abandon", "quit", "q"):
            return "abandon"
        # On tente d'abord la notation courante, puis l'UCI.
        try:
            return board.parse_san(txt)
        except ValueError:
            pass
        try:
            m = chess.Move.from_uci(bas)
            if m in board.legal_moves:
                return m
            print(f"  {txt} n'est pas légal ici. Tape ? pour la liste.")
        except ValueError:
            print(f"  {txt} n'est ni une notation courante ni un coup UCI.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/run1_best.pt")
    p.add_argument("--vocab", default="data/vocab.json")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--temperature", type=float, default=0.0,
                   help="0 = le modèle joue toujours son meilleur coup")
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--noir", action="store_true", help="tu joues les noirs")
    p.add_argument("--analyse", action="store_true",
                   help="montre l'avis du modèle à chaque coup")
    args = p.parse_args()

    eng = ChessEngine(args.ckpt, args.vocab, args.device,
                      args.temperature, args.top_k)
    humain_blanc = not args.noir

    print(f"\n  Modèle : {os.path.basename(args.ckpt)} "
          f"(step {eng.ckpt_step:,}, {eng.tokens_seen/1e6:.0f} M tokens vus)")
    print(f"  Tu joues les {'noirs' if args.noir else 'blancs'}. "
          f"Tape ? pour l'aide.\n")

    board = chess.Board()
    histoire: list[str] = []

    while not board.is_game_over(claim_draw=True):
        tour_humain = (board.turn == chess.WHITE) == humain_blanc
        afficher(board, humain_blanc)

        if tour_humain:
            if args.analyse:
                eng.set_position(board, histoire)
                afficher_analyse(eng, board)
            coup = lire_coup(board)
            if coup == "abandon":
                print("  Tu abandonnes.")
                return
            if coup == "analyse":
                eng.set_position(board, histoire)
                afficher_analyse(eng, board, n=10)
                continue
            if coup == "annuler":
                # Deux demi-coups : le sien et celui du modèle.
                for _ in range(2):
                    if board.move_stack:
                        board.pop()
                        histoire.pop()
                continue
            board.push(coup)
            histoire.append(coup.uci())
        else:
            eng.set_position(board, histoire)
            coup = eng.best_move(board)
            san = board.san(coup)
            board.push(coup)
            histoire.append(coup.uci())
            print(f"  Le modèle joue : {san}  ({coup.uci()})")

    afficher(board, humain_blanc)
    issue = board.outcome(claim_draw=True)
    if issue.winner is None:
        print(f"  Partie nulle, {issue.termination.name.lower()}")
    else:
        gagnant = "toi" if (issue.winner == chess.WHITE) == humain_blanc \
                  else "le modèle"
        print(f"  {gagnant.capitalize()} gagne, "
              f"{issue.termination.name.lower()}")
    print(f"\n  {len(histoire)} demi-coups : {' '.join(histoire)}\n")


if __name__ == "__main__":
    main()
