"""Construction du vocabulaire : un token = un coup UCI.

C'est la décision de conception la plus structurante du projet, alors elle
mérite d'être expliquée.

Un modèle de langage classique découpe le texte en fragments de mots (BPE).
Ici on fait autrement : le vocabulaire est l'ensemble *fini et connu d'avance*
de tous les coups jouables aux échecs, écrits en notation UCI, case de départ
suivie de la case d'arrivée, plus éventuellement la pièce de promotion :
`e2e4`, `g1f3`, `e7e8q`. Un coup = un token, indivisible.

Pourquoi c'est mieux qu'un BPE ici : avec un BPE, `e2e4` pourrait se découper
en `e2` + `e4`, ou pire en `e` + `2e` + `4`, et le modèle devrait dépenser de
la capacité à réapprendre que ces fragments forment une unité. Avec un token
par coup, une partie de 80 demi-coups fait exactement 80 tokens, la fenêtre de
contexte se raisonne en coups plutôt qu'en caractères, et le masquage des
coups illégaux (phase 4) devient un simple masque booléen sur le vocabulaire.

Comment on énumère les coups possibles : plutôt que d'essayer de deviner quels
coups sont légaux (ça dépend de la position), on énumère tous les déplacements
*géométriquement* possibles sur un échiquier vide. Toute pièce se déplace soit
comme une dame (lignes, colonnes, diagonales), soit comme un cavalier, le roi,
la tour, le fou et le pion ne font que des sous-ensembles des déplacements de
la dame. On ajoute ensuite les promotions. Le vocabulaire obtenu est un
sur-ensemble strict des coups légaux, ce qui est exactement ce qu'on veut :
il ne manquera jamais un coup, et le modèle apprendra tout seul que certains
n'apparaissent jamais.
"""

import json

FILES = "abcdefgh"
RANKS = "12345678"

SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>"]

# Déplacements du cavalier, en (delta_colonne, delta_ligne)
KNIGHT_DELTAS = [(1, 2), (2, 1), (2, -1), (1, -2),
                 (-1, -2), (-2, -1), (-2, 1), (-1, 2)]

# Directions de la dame : les 8 rayons (horizontal, vertical, diagonal)
QUEEN_DIRECTIONS = [(0, 1), (1, 1), (1, 0), (1, -1),
                    (0, -1), (-1, -1), (-1, 0), (-1, 1)]

PROMOTION_PIECES = "qrbn"


def square_name(file_idx: int, rank_idx: int) -> str:
    """(0, 0) -> 'a1', (7, 7) -> 'h8'."""
    return FILES[file_idx] + RANKS[rank_idx]


def build_move_list() -> list[str]:
    """Énumère tous les coups UCI géométriquement possibles.

    Renvoie une liste triée, donc déterministe : le même vocabulaire sera
    reconstruit à l'identique sur n'importe quelle machine, ce qui évite les
    désastres silencieux du type « le checkpoint a été entraîné avec un autre
    ordre de tokens ».
    """
    moves = set()

    for from_file in range(8):
        for from_rank in range(8):
            origin = square_name(from_file, from_rank)

            # Déplacements de dame : on parcourt chaque rayon jusqu'au bord
            for d_file, d_rank in QUEEN_DIRECTIONS:
                for distance in range(1, 8):
                    to_file = from_file + d_file * distance
                    to_rank = from_rank + d_rank * distance
                    if not (0 <= to_file < 8 and 0 <= to_rank < 8):
                        break
                    moves.add(origin + square_name(to_file, to_rank))

            # Déplacements de cavalier
            for d_file, d_rank in KNIGHT_DELTAS:
                to_file = from_file + d_file
                to_rank = from_rank + d_rank
                if 0 <= to_file < 8 and 0 <= to_rank < 8:
                    moves.add(origin + square_name(to_file, to_rank))

    # Promotions. Un pion blanc promeut en allant de la 7e à la 8e rangée,
    # un pion noir de la 2e à la 1re. Il peut avancer tout droit ou capturer
    # en diagonale, d'où le décalage de colonne dans {-1, 0, +1}.
    for from_rank, to_rank in ((6, 7), (1, 0)):
        for from_file in range(8):
            for d_file in (-1, 0, 1):
                to_file = from_file + d_file
                if not 0 <= to_file < 8:
                    continue
                base = (square_name(from_file, from_rank)
                        + square_name(to_file, to_rank))
                for piece in PROMOTION_PIECES:
                    moves.add(base + piece)

    return sorted(moves)


def build_vocab() -> dict:
    """Construit le dictionnaire complet : tokens spéciaux puis coups."""
    move_list = build_move_list()
    itos = SPECIAL_TOKENS + move_list
    stoi = {token: idx for idx, token in enumerate(itos)}

    assert len(itos) == len(stoi), "collision dans le vocabulaire"
    # Le vocabulaire doit tenir sur 16 bits, puisqu'on encode le corpus en
    # uint16 pour diviser par deux la taille du fichier .bin.
    assert len(itos) < 2 ** 16, "vocabulaire trop grand pour un uint16"

    return {
        "itos": itos,
        "stoi": stoi,
        "vocab_size": len(itos),
        "n_special": len(SPECIAL_TOKENS),
        "n_moves": len(move_list),
        "pad_id": stoi["<pad>"],
        "bos_id": stoi["<bos>"],
        "eos_id": stoi["<eos>"],
    }


def save_vocab(path: str) -> dict:
    vocab = build_vocab()
    with open(path, "w") as f:
        json.dump(vocab, f, indent=2)
    return vocab


if __name__ == "__main__":
    import sys

    out = sys.argv[1] if len(sys.argv) > 1 else "data/vocab.json"
    v = save_vocab(out)
    print(f"Vocabulaire écrit dans {out}")
    print(f"  taille totale      : {v['vocab_size']}")
    print(f"  tokens spéciaux    : {v['n_special']} ({', '.join(SPECIAL_TOKENS)})")
    print(f"  coups UCI distincts: {v['n_moves']}")
    print(f"  exemples           : {', '.join(v['itos'][3:9])} ... "
          f"{', '.join(v['itos'][-4:])}")
