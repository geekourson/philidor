"""Un moteur d'échecs UCI standard, autour du modèle.

Deux usages :

    # comme moteur en ligne de commande (protocole UCI, pour cutechess/lichess-bot)
    python engine.py --ckpt checkpoints/run1_best.pt

    # comme bibliothèque
    from engine import ChessEngine
    eng = ChessEngine("checkpoints/run1_best.pt", device="cuda:0")
    move = eng.best_move(board)

Le point important est le **masque de légalité**. Le modèle, laissé libre,
propose parfois un coup impossible : c'est précisément ce que mesure la phase
4. Mais pour *jouer*, un tel coup est disqualifiant : un moteur UCI qui répond
un coup illégal perd la partie sur-le-champ.

La solution retenue est la plus simple possible : un seul passage avant, puis
on met à moins l'infini les logits de tous les tokens qui ne correspondent pas
à un coup légal dans la position courante, avant de prendre le maximum. Le
modèle choisit donc toujours parmi les coups légaux, en conservant l'ordre de
préférence qu'il avait appris.

On aurait pu faire autrement, rééchantillonner jusqu'à tomber sur un coup
légal, ou faire une recherche arborescente. Le masque a été préféré parce
qu'il est déterministe, qu'il coûte exactement un passage avant, et qu'il
n'altère pas les préférences relatives du modèle entre coups légaux. C'était
aussi le critère annoncé : à qualité comparable, on choisit ce qui s'explique
le plus simplement.
"""

import argparse
import json
import os
import sys

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import chess
import torch
import torch.nn.functional as F

from model import ChessGPT, ModelConfig


class ChessEngine:

    def __init__(self, ckpt_path: str, vocab_path: str = "data/vocab.json",
                 device: str = "cuda:0", temperature: float = 0.0,
                 top_k: int | None = None):
        self.device = device
        self.temperature = temperature
        self.top_k = top_k

        with open(vocab_path) as f:
            v = json.load(f)
        self.stoi = v["stoi"]
        self.itos = v["itos"]
        self.bos = v["bos_id"]
        self.eos = v["eos_id"]
        self.vocab_size = v["vocab_size"]

        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        cfg = ModelConfig(**ck["model_config"])
        self.model = ChessGPT(cfg).to(device)
        self.model.load_state_dict(ck["model"])
        self.model.eval()
        self.block_size = cfg.block_size
        self.ckpt_step = ck.get("step", 0)
        self.tokens_seen = ck.get("tokens_seen", 0)

        # Le modèle ne voit que la liste des coups depuis le début de la
        # partie. On la maintient ici en parallèle de l'échiquier.
        self.history: list[int] = []

    # -- gestion de la position ---------------------------------------------

    def set_position(self, board: chess.Board, move_history: list[str]):
        """Renseigne la suite de coups joués depuis la position initiale.

        Le modèle n'a aucune représentation de l'échiquier : il ne comprend
        qu'une séquence de coups. Une position transmise sous forme de FEN sans
        historique lui est donc inutilisable, d'où la limitation documentée
        plus bas dans `uci_loop`.
        """
        self.history = [self.stoi[m] for m in move_history if m in self.stoi]

    # -- le cœur -------------------------------------------------------------

    @torch.no_grad()
    def move_scores(self, board: chess.Board):
        """Renvoie {coup UCI: probabilité} restreint aux coups légaux."""
        ids = [self.bos] + self.history
        ids = ids[-self.block_size:]
        x = torch.tensor([ids], dtype=torch.long, device=self.device)

        with torch.autocast("cuda", dtype=torch.bfloat16) \
                if self.device.startswith("cuda") else _null():
            logits, _ = self.model(x)
        logits = logits[0, -1, :].float()

        # Le masque : un booléen par entrée du vocabulaire.
        mask = torch.zeros(self.vocab_size, dtype=torch.bool,
                           device=self.device)
        legal_moves = list(board.legal_moves)
        for move in legal_moves:
            uci = move.uci()
            idx = self.stoi.get(uci)
            if idx is not None:
                mask[idx] = True

        if not mask.any():
            # Ne devrait pas arriver : le vocabulaire est un sur-ensemble des
            # coups légaux. Si ça arrive, mieux vaut un coup légal au hasard
            # qu'un plantage en pleine partie.
            return {m.uci(): 1.0 / len(legal_moves) for m in legal_moves}

        logits = logits.masked_fill(~mask, float("-inf"))
        probs = F.softmax(logits, dim=-1)
        return {self.itos[i]: float(probs[i])
                for i in torch.nonzero(mask).flatten().tolist()}

    @torch.no_grad()
    def best_move(self, board: chess.Board) -> chess.Move:
        scores = self.move_scores(board)
        if self.temperature == 0.0:
            uci = max(scores, key=scores.get)
        else:
            items = list(scores.items())
            logits = torch.tensor([s for _, s in items]).log() / self.temperature
            if self.top_k:
                k = min(self.top_k, len(items))
                keep = logits.topk(k).indices
                filt = torch.full_like(logits, float("-inf"))
                filt[keep] = logits[keep]
                logits = filt
            i = int(torch.multinomial(F.softmax(logits, dim=-1), 1))
            uci = items[i][0]
        return chess.Move.from_uci(uci)

    # -- protocole UCI -------------------------------------------------------

    def uci_loop(self):
        """Boucle du protocole UCI, lue sur stdin, réponses sur stdout.

        Limitation assumée et importante : la commande `position fen ...` n'est
        pas réellement supportée, parce que le modèle a besoin de l'historique
        des coups et non d'une position. On accepte `position startpos moves
        ...`, qui est ce qu'envoient cutechess-cli et lichess-bot en jeu
        normal. Sur un `position fen`, le moteur joue quand même un coup légal
       , il se rabat sur le coup le plus probable *sans historique*, ce qui
        revient à jouer comme en début de partie. Le comportement est dégradé
        mais jamais illégal.
        """
        board = chess.Board()
        history: list[str] = []

        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            cmd = parts[0]

            if cmd == "uci":
                print(f"id name ChessGPT-{self.ckpt_step}")
                print("id author Billy Girboux")
                print("uciok", flush=True)

            elif cmd == "isready":
                print("readyok", flush=True)

            elif cmd == "ucinewgame":
                board = chess.Board()
                history = []
                self.history = []

            elif cmd == "position":
                if "startpos" in parts:
                    board = chess.Board()
                    history = []
                    if "moves" in parts:
                        for uci in parts[parts.index("moves") + 1:]:
                            board.push(chess.Move.from_uci(uci))
                            history.append(uci)
                elif "fen" in parts:
                    i = parts.index("fen")
                    end = parts.index("moves") if "moves" in parts else len(parts)
                    board = chess.Board(" ".join(parts[i + 1:end]))
                    history = []
                    if "moves" in parts:
                        for uci in parts[parts.index("moves") + 1:]:
                            board.push(chess.Move.from_uci(uci))
                            history.append(uci)
                self.set_position(board, history)

            elif cmd == "go":
                move = self.best_move(board)
                print(f"bestmove {move.uci()}", flush=True)

            elif cmd in ("quit", "stop"):
                if cmd == "quit":
                    break

            elif cmd == "setoption":
                pass   # aucune option exposée pour l'instant


class _null:
    def __enter__(self): return None
    def __exit__(self, *a): return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--vocab", default="data/vocab.json")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--selftest", action="store_true",
                   help="joue une partie contre soi-même et vérifie la légalité")
    args = p.parse_args()

    eng = ChessEngine(args.ckpt, args.vocab, args.device,
                      args.temperature, args.top_k)

    if args.selftest:
        board = chess.Board()
        history = []
        while not board.is_game_over() and len(history) < 300:
            eng.set_position(board, history)
            move = eng.best_move(board)
            assert move in board.legal_moves, f"coup illégal produit : {move}"
            board.push(move)
            history.append(move.uci())
        print(f"partie complète : {len(history)} demi-coups, "
              f"résultat {board.result()}, aucun coup illégal")
        print(" ".join(history[:40]) + (" ..." if len(history) > 40 else ""))
        return

    eng.uci_loop()


if __name__ == "__main__":
    main()
