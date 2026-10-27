"""Étape A en partie : recherche PUCT, la politique guide, la valeur note.

Même recherche que recherche_verrou.py (recherche.py), pour une seule position :
`sims` évaluations de feuilles, par lots de `lot` grâce à la perte virtuelle.
Le plateau reçu doit porter l'historique (pile de coups) pour que la triple
répétition soit détectée dans l'arbre.

`sims = 0` revient au moteur politique + valeur sans recherche (top-k,
argmax(note + alpha log p)) ; `sims = -1` à la politique seule, une passe. Le
serveur d'inférence règle `sims` à chaque coup selon la pendule
(budget_pendule.py).
"""
import json
from collections import OrderedDict
import chess
from engine import ChessEngine
from juge_recherche.engine_valeur import ValeurEngine
from juge_recherche.recherche import Recherche, chercher, sous_arbre


class RechercheEngine(ValeurEngine):
    def __init__(self, ckpt_path, vocab_path, device="cuda:0", temperature=0.0,
                 sims=64, cpuct=0.1, fpu=0.0, k=5, lot=8, alpha=0.25,
                 valeur_path="checkpoints/valeur.pt", graphes=False, lot_auto=False, k_recherche=None,
                 reprise=False):
        super().__init__(ckpt_path, vocab_path, device, temperature, k=k, alpha=alpha,
                         valeur_path=valeur_path)
        self.sims, self.cpuct, self.fpu, self.lot = sims, cpuct, fpu, lot
        # largeur de l'arbre (coups examinés par position) ; self.k reste celle du
        # juge seul (sims = 0), qu'on ne change pas sans l'avoir mesurée
        self.k_recherche = k_recherche or k
        self.pad = json.load(open(vocab_path))["pad_id"]
        self.derniere = None
        # reprise : on garde l'arbre de chaque partie et on repart, au coup suivant,
        # du nœud atteint par notre coup et la réponse de l'adversaire
        self.reprise = reprise
        self.arbres = OrderedDict()    # historique (jetons) -> recherche ; plusieurs parties à la fois
        self.arret = None     # le serveur y branche « une autre partie attend »
        self.lot_auto = lot_auto   # lot 8 / 16 / 32 selon le nombre de simulations
        self.evaluateur = None
        if graphes:                # graphes CUDA : une passe rejouée d'un seul appel
            from juge_recherche.graphes import Evaluateur
            self.evaluateur = Evaluateur(self.model, self.valeur, self.pad, device,
                                         tailles_lot=(8, 16, 32) if lot_auto else (lot,))

    def best_move(self, board: chess.Board) -> chess.Move:
        if self.sims < 0:                     # pendule presque vide : une passe, l'intuition seule
            return ChessEngine.best_move(self, board)
        if self.sims == 0:                    # deux passes : politique + juge
            return ValeurEngine.best_move(self, board)
        legaux = list(board.legal_moves)
        if len(legaux) == 1:
            return legaux[0]
        racine = None
        if self.reprise:
            ancienne = self.arbres.pop(tuple(self.history[:-2]), None)
            racine = sous_arbre(ancienne, self.history[-2:]) if ancienne is not None else None
        r = Recherche(board, [self.bos] + list(self.history), self.stoi,
                      k=self.k_recherche, cpuct=self.cpuct, fpu=self.fpu, ctx=self.block_size, racine=racine)
        lot = (8 if self.sims <= 64 else 16 if self.sims <= 256 else 32) if self.lot_auto else self.lot
        chercher([r], self.model, self.valeur, self.sims, self.pad, self.device, lot=lot,
                 arret=self.arret, evaluateur=self.evaluateur)
        self.derniere = r
        if self.reprise:
            self.arbres[tuple(self.history)] = r
            while len(self.arbres) > 32:           # parties abandonnées ou terminées
                self.arbres.popitem(last=False)
        return r.choix()
