"""Étape 0 : Stockfish choisit parmi le top-k de Philidor.

Ce n'est PAS un bot à déployer : Stockfish y fait tout le travail de jugement.
C'est un instrument de mesure. Il chiffre en Elo le plafond trouvé par la
mesure 1 du diagnostic : le meilleur coup est dans le top-5 du modèle dans
71,5 % de ses gaffes, et un sélecteur parfait ramènerait le taux de gaffes de
18,32 % à 5,22 %. Combien de points d'Elo valent ces 13 points de gaffes ?
La réponse décide s'il vaut la peine de construire un juge appris.

Le modèle propose (une passe avant, masque de légalité, top-k), Stockfish
tranche entre ces k coups et eux seuls (`root_moves`), à profondeur fixe.

DÉTERMINISME. Stockfish garde sa table de hachage d'un coup à l'autre, et son
contenu dépend de l'historique des appels : deux instances qui voient les mêmes
positions dans un ordre différent peuvent jouer différemment. Le témoin clone
le détecterait. On passe donc un objet `game` neuf à chaque coup, ce qui fait
envoyer `ucinewgame` et vider la table : chaque décision devient une fonction
pure de la position.
"""

import os
import chess
import chess.engine

from engine import ChessEngine

SF_BIN = os.environ.get("STOCKFISH", "stockfish")


class SFSelectEngine(ChessEngine):
    def __init__(self, ckpt_path, vocab_path, device="cuda:0",
                 temperature=0.0, k=5, depth=4, sf_bin=SF_BIN):
        super().__init__(ckpt_path, vocab_path, device, temperature)
        self.k, self.depth = k, depth
        self.sf = chess.engine.SimpleEngine.popen_uci(sf_bin)
        self.sf.configure({"Threads": 1, "Hash": 16})

    def best_move(self, board: chess.Board) -> chess.Move:
        scores = self.move_scores(board)          # une passe, sans bourrage
        top = sorted(scores, key=scores.get, reverse=True)[:self.k]
        if len(top) == 1:
            return chess.Move.from_uci(top[0])
        r = self.sf.play(board, chess.engine.Limit(depth=self.depth),
                         root_moves=[chess.Move.from_uci(u) for u in top],
                         game=object())
        return r.move

    def __del__(self):
        try:
            self.sf.quit()
        except Exception:
            pass
