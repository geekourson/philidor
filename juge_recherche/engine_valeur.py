"""Étape 1 bis en partie : la politique propose, le réseau de valeur tranche.

Deux réseaux de même taille : run2 (la politique, inchangée) donne le top-k et
les log-probabilités ; la copie affinée en valeur (valeur_train.py) note la
position obtenue après chaque candidat, vue par le joueur au trait. Les k
séquences « préfixe + candidat » ont la même longueur : lot rectangulaire, aucun
bourrage. On joue argmax(note + alpha x log p).

Pas de Stockfish, pas de recherche : deux passes avant par coup.
"""
import math
import chess
import torch
from engine import ChessEngine, _null
from model import ChessGPT, ModelConfig
from juge_recherche.valeur_train import Valeur


class ValeurEngine(ChessEngine):
    def __init__(self, ckpt_path, vocab_path, device="cuda:0", temperature=0.0, k=5, alpha=0.5,
                 valeur_path="checkpoints/valeur.pt"):
        super().__init__(ckpt_path, vocab_path, device, temperature)
        self.k, self.alpha = k, alpha
        ck = torch.load(valeur_path, map_location="cpu", weights_only=False)
        v = Valeur(ChessGPT(ModelConfig(**ck["model_config"])))
        v.load_state_dict(ck["modele"])
        self.valeur = v.to(device).eval()

    @torch.no_grad()
    def best_move(self, board: chess.Board) -> chess.Move:
        scores = self.move_scores(board)
        top = sorted(scores, key=scores.get, reverse=True)[:self.k]
        if len(top) == 1:
            return chess.Move.from_uci(top[0])
        base = [self.bos] + list(self.history)
        x = torch.tensor([(base + [self.stoi[u]])[-self.block_size:] for u in top],
                         dtype=torch.long, device=self.device)
        with torch.autocast("cuda", dtype=torch.bfloat16) \
                if self.device.startswith("cuda") else _null():
            note = self.valeur(x)[:, -1].float().cpu().tolist()
        total = [n + self.alpha * math.log(max(scores[u], 1e-9)) for n, u in zip(note, top)]
        return chess.Move.from_uci(top[max(range(len(top)), key=total.__getitem__)])
