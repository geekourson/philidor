"""Distribution complète du modèle sur les coups légaux, température 0.

Une passe avant par position. Les logits sont masqués aux coups légaux comme en
partie, puis normalisés : on obtient la probabilité de chaque coup légal, donc
le classement complet du modèle, pas seulement son argmax.

Sert aux mesures 1 (proposition contre sélection) et 4 (lookahead).

    python diag_modele.py --ckpt checkpoints/run2_best.pt --device cuda:0
"""

import argparse
import json
import os
import time

import chess
import torch
import torch.nn.functional as F

from model import ChessGPT, ModelConfig


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/run2_best.pt")
    p.add_argument("--vocab", default="data/vocab.json")
    p.add_argument("--positions", default="diag/positions.jsonl")
    p.add_argument("--out", default="diag/modele.jsonl")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--lot", type=int, default=64)
    args = p.parse_args()

    with open(args.vocab) as f:
        v = json.load(f)
    stoi, itos, bos = v["stoi"], v["itos"], v["bos_id"]

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ModelConfig(**ck["model_config"])
    model = ChessGPT(cfg)
    model.load_state_dict(ck["model"])
    model = model.to(args.device).eval()
    print(f"[modele] {args.ckpt} | {sum(q.numel() for q in model.parameters())/1e6:.1f} M",
          flush=True)

    lignes = open(args.positions).read().splitlines()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    t0 = time.time()
    n = 0
    with open(args.out, "w") as g:
        # UNE POSITION A LA FOIS, sans bourrage.
        #
        # La premiere version mettait les positions en lot avec un bourrage a
        # GAUCHE par <bos>. Le modele n'a aucun masque d'attention pour le
        # bourrage : il voyait donc "<bos> <bos> ... <bos> e2e4 ..." au lieu de
        # "<bos> e2e4 ...". Hors distribution. Verifie sur 200 positions : le
        # lot ne donnait le meme coup que la position seule que 70 fois sur
        # 200, quand le moteur de production la donne 200 fois sur 200.
        # 5 000 positions prennent deux minutes ainsi, ce n'est pas un sujet.
        for k in range(len(lignes)):
            lot = [json.loads(lignes[k])]
            ids = [bos] + [stoi[m] for m in lot[0]["prefixe"] if m in stoi]
            x = torch.tensor([ids[-cfg.block_size:]], dtype=torch.long)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                out, _ = model(x.to(args.device))
            lg = out[:, -1, :].float().cpu()

            for i, d in enumerate(lot):
                b = chess.Board(d["fen"])
                masque = torch.zeros(cfg.vocab_size, dtype=torch.bool)
                idx = {}
                for mv in b.legal_moves:
                    j = stoi.get(mv.uci())
                    if j is not None:
                        masque[j] = True
                        idx[j] = mv.uci()
                probs = F.softmax(lg[i].masked_fill(~masque, float("-inf")), -1)
                ordre = sorted(idx, key=lambda j: -float(probs[j]))
                g.write(json.dumps({
                    "fen": d["fen"],
                    "classement": [idx[j] for j in ordre],
                    "p": [round(float(probs[j]), 6) for j in ordre],
                }, separators=(",", ":")) + "\n")
                n += 1
            if n % 1000 == 0:
                print(f"  {n:,}/{len(lignes):,} ({time.time()-t0:.0f}s)",
                      flush=True)

    print(f"\n{n:,} positions -> {args.out}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
