"""Étape 1 du Q : quels coups le modèle préfère-t-il, et où se trompe-t-il ?

Pour chaque position, on récupère les k coups que le modèle joue le plus
volontiers. Stockfish évaluera ensuite ces coups-là **en plus** de ses propres
meilleurs, ce qui est le point crucial.

Le diagnostic du 7 septembre est sans ambiguïté : 93,2 % des défaites tiennent à
une gaffe unique. Un Q entraîne uniquement sur les cinq meilleurs coups de
Stockfish apprendrait a classer entre bons coups et ne verrait **jamais** un
coup catastrophique note. Il faut lui montrer ce que valent ses propres erreurs.

Et on n'echantillonne qu'a partir du demi-coup 20 : l'ouverture est jouee a
19 centipions pres avec 0,6 % de gaffes, l'etiqueter serait payer Stockfish pour
enseigner ce qui est deja acquis.

Sortie JSONL : {"prefixe": [...], "candidats_modele": ["e2e4", ...]}
"""

import argparse
import json
import os
import random
import time
from collections import defaultdict

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import chess
import torch

from model import ChessGPT, ModelConfig


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/run2_best.pt")
    p.add_argument("--vocab", default="data/vocab.json")
    p.add_argument("--games", default="data/games_uci.txt")
    p.add_argument("--out", required=True)
    p.add_argument("--positions", type=int, default=500_000)
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--min-ply", type=int, default=20)
    p.add_argument("--lot", type=int, default=256)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=11)
    args = p.parse_args()

    with open(args.vocab) as f:
        v = json.load(f)
    stoi, itos, bos = v["stoi"], v["itos"], v["bos_id"]
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ModelConfig(**ck["model_config"])
    model = ChessGPT(cfg)
    model.load_state_dict(ck["model"])
    model.to(args.device).eval()
    print(f"[modele] {args.ckpt}", flush=True)

    rng = random.Random(args.seed)
    fh = open(args.out, "w")
    seaux = defaultdict(list)
    n = 0
    t0 = time.perf_counter()

    @torch.no_grad()
    def traiter(k, lot):
        nonlocal n
        boards, pre = [], []
        for coups in lot:
            b = chess.Board()
            ok = True
            for m in coups[:k]:
                try:
                    b.push(chess.Move.from_uci(m))
                except Exception:
                    ok = False
                    break
            if not ok or b.is_game_over():
                continue
            boards.append(b)
            pre.append([bos] + [stoi[m] for m in coups[:k] if m in stoi])
        if not boards:
            return
        x = torch.tensor([s[-cfg.block_size:] for s in pre],
                         dtype=torch.long, device=args.device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out, _ = model(x)
        lg = out[:, -1, :].float()
        m = torch.zeros(len(boards), cfg.vocab_size, dtype=torch.bool,
                        device=args.device)
        for i, b in enumerate(boards):
            for mv in b.legal_moves:
                j = stoi.get(mv.uci())
                if j is not None:
                    m[i, j] = True
        lg = lg.masked_fill(~m, float("-inf"))
        top = lg.topk(min(args.k, lg.shape[1]), dim=-1).indices.tolist()
        for b, s, t in zip(boards, pre, top):
            legaux = {mv.uci() for mv in b.legal_moves}
            cands = [itos[j] for j in t if itos[j] in legaux]
            if not cands:
                continue
            fh.write(json.dumps({"prefixe": [itos[j] for j in s[1:]],
                                 "candidats_modele": cands}) + "\n")
            n += 1

    with open(args.games) as f:
        for ligne in f:
            if n >= args.positions:
                break
            c = ligne.split()
            if len(c) < args.min_ply + 4:
                continue
            k = rng.randrange(args.min_ply, len(c))
            seaux[k].append(c)
            if len(seaux[k]) >= args.lot:
                traiter(k, seaux.pop(k))
                if n % 50_000 < args.lot:
                    dt = time.perf_counter() - t0
                    print(f"  {n:,}/{args.positions:,} positions, "
                          f"{dt/60:.1f} min ({n/max(dt,1):.0f}/s)", flush=True)
    for k in list(seaux):
        traiter(k, seaux.pop(k))
    fh.close()
    print(f"\n{n:,} positions -> {args.out}")
    print(f"duree : {(time.perf_counter()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
