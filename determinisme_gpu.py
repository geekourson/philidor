"""Deux cartes, mêmes poids, aucune source de hasard : jouent-elles pareil ?

Le moteur est réputé déterministe. Température 0, donc argmax. Poids figés,
graine fixée, ouvertures identiques. Et pourtant le même duel rejoué sur une
autre carte donne un écart d'Elo différent de cinquante points.

On mesure d'où ça vient, position par position :

- la fréquence à laquelle les deux cartes ne choisissent PAS le même coup ;
- l'écart numérique entre leurs logits ;
- et surtout la **marge** entre le premier et le deuxième coup dans les cas de
  désaccord, qui doit être minuscule si l'explication est la précision.

Le tout en bf16 (ce que fait le moteur en production) puis en fp32, pour savoir
si c'est la demi-précision qui est en cause ou l'ordre des opérations CUDA.

Usage :
    python determinisme_gpu.py --ckpt checkpoints/run2_best.pt --n 2000
"""

import argparse
import json
import os
import random

import chess
import torch
import torch.nn.functional as F

from model import ChessGPT, ModelConfig


def charger(ckpt, device):
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = ModelConfig(**ck["model_config"])
    m = ChessGPT(cfg)
    m.load_state_dict(ck["model"])
    return m.to(device).eval(), cfg


@torch.no_grad()
def logits_position(model, cfg, ids, device, bf16):
    x = torch.tensor([ids[-cfg.block_size:]], dtype=torch.long, device=device)
    if bf16:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out, _ = model(x)
    else:
        out, _ = model(x)
    return out[0, -1, :].float().cpu()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/run2_best.pt")
    p.add_argument("--vocab", default="data/vocab.json")
    p.add_argument("--games", default="data/val_games.txt")
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--a", default="cuda:0")
    p.add_argument("--b", default="cuda:1")
    p.add_argument("--out", default="results/determinisme_gpu.json")
    args = p.parse_args()

    with open(args.vocab) as f:
        v = json.load(f)
    stoi, itos, bos = v["stoi"], v["itos"], v["bos_id"]

    ma, cfg = charger(args.ckpt, args.a)
    mb, _ = charger(args.ckpt, args.b)
    print(f"[cartes] A={args.a}  B={args.b}", flush=True)
    for d in (args.a, args.b):
        print(f"  {d} : {torch.cuda.get_device_name(d)}")

    rng = random.Random(4)
    positions = []
    for ligne in open(args.games):
        c = ligne.split()
        if len(c) < 20:
            continue
        positions.append(c[:rng.randrange(8, min(len(c), 100))])
        if len(positions) >= args.n:
            break

    rapport = {}
    for bf16 in (True, False):
        nom = "bf16" if bf16 else "fp32"
        desaccords = 0
        ecarts_logit = []
        marges_desaccord = []
        marges_accord = []
        for coups in positions:
            board = chess.Board()
            for m in coups:
                board.push(chess.Move.from_uci(m))
            ids = [bos] + [stoi[m] for m in coups if m in stoi]
            la = logits_position(ma, cfg, ids, args.a, bf16)
            lb = logits_position(mb, cfg, ids, args.b, bf16)

            masque = torch.zeros(cfg.vocab_size, dtype=torch.bool)
            for mv in board.legal_moves:
                j = stoi.get(mv.uci())
                if j is not None:
                    masque[j] = True
            if not masque.any():
                continue
            la = la.masked_fill(~masque, float("-inf"))
            lb = lb.masked_fill(~masque, float("-inf"))

            ecarts_logit.append(float((la[masque] - lb[masque]).abs().max()))
            ta = la.topk(min(2, int(masque.sum())))
            marge = float(ta.values[0] - ta.values[1]) if len(ta.values) > 1 else 99.
            if int(la.argmax()) != int(lb.argmax()):
                desaccords += 1
                marges_desaccord.append(marge)
            else:
                marges_accord.append(marge)

        n = len(ecarts_logit)
        med = lambda t: sorted(t)[len(t)//2] if t else float("nan")
        r = {
            "positions": n,
            "desaccords": desaccords,
            "taux_desaccord": desaccords / max(n, 1),
            "ecart_logit_median": med(ecarts_logit),
            "ecart_logit_max": max(ecarts_logit) if ecarts_logit else 0,
            "marge_mediane_si_desaccord": med(marges_desaccord),
            "marge_mediane_si_accord": med(marges_accord),
        }
        rapport[nom] = r
        print(f"\n=== {nom} ===")
        print(f"positions comparees            : {n:,}")
        print(f"coups differents entre cartes  : {desaccords} "
              f"({r['taux_desaccord']:.2%})")
        print(f"ecart de logit, mediane        : {r['ecart_logit_median']:.2e}")
        print(f"ecart de logit, maximum        : {r['ecart_logit_max']:.2e}")
        print(f"marge 1er/2e SI desaccord      : {r['marge_mediane_si_desaccord']:.4f}")
        print(f"marge 1er/2e si accord         : {r['marge_mediane_si_accord']:.4f}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(rapport, f, indent=2)
    print(f"\nrapport : {args.out}")


if __name__ == "__main__":
    main()
