"""Étape 1 en partie : la politique propose, le juge tranche.

Une passe avant pour la politique (top-k et log-probabilités), puis UNE passe par
lot sur les k séquences « préfixe + candidat ». Elles ont toutes la même
longueur, donc le lot est rectangulaire sans aucun bourrage, l'erreur du
diagnostic ne peut pas se reproduire ici. On lit le bloc 15 et la sortie au
dernier jeton, le juge note, et on joue argmax(juge + alpha x log p).

Pas de Stockfish, pas de recherche : deux passes avant par coup.
"""
import math
import chess
import torch
from engine import ChessEngine, _null
from juge_recherche.juge_train import Juge


class JugeEngine(ChessEngine):
    def __init__(self, ckpt_path, vocab_path, device="cuda:0", temperature=0.0,
                 k=5, alpha=0.5,
                 juge_path="checkpoints/juge.pt"):
        super().__init__(ckpt_path, vocab_path, device, temperature)
        self.k, self.alpha = k, alpha
        ck = torch.load(juge_path, map_location="cpu", weights_only=False)
        self.juge = Juge(ck["d"]); self.juge.load_state_dict(ck["etat"])
        self.juge = self.juge.to(device).eval()
        self._e = {}
        self.model.blocks[14].register_forward_hook(lambda m, i, o: self._e.__setitem__("b", o))
        self.model.norm_final.register_forward_hook(lambda m, i, o: self._e.__setitem__("f", o))

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
            self.model(x)
        h = torch.cat([self._e["b"][:, -1, :], self._e["f"][:, -1, :]], -1).float()
        note = self.juge(h).cpu().tolist()
        total = [n + self.alpha * math.log(max(scores[u], 1e-9)) for n, u in zip(note, top)]
        return chess.Move.from_uci(top[max(range(len(top)), key=total.__getitem__)])
