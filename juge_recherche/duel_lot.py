"""Duel en lot : des centaines de parties jouées en même temps.

Même protocole que elo_match.py (ouvertures appariées tirées par
`tirer_ouverture`, couleurs alternées, fin de partie `claim_draw=True`, limite
de 400 demi-coups, Elo par `elo_with_error`), mais toutes les parties avancent
ensemble : à chaque tour, chaque joueur calcule son coup pour TOUTES les
parties où il a le trait, en lots sur le GPU.

Pourquoi : un joueur à recherche lance des milliers de petites opérations par
coup (lots de 8 feuilles) ; seul sur la 3090 il prend 660 ms par coup, et deux
processus sur une même carte se ralentissent à tour de rôle (3 s par coup). En
lot, 128 recherches avancent ensemble (176 ms par position hors ligne).

Joueurs :
    base                         argmax de la politique (coups légaux)
    valeur:k:alpha               top-k de la politique, argmax(note + alpha log p)
                                 (juge v2 : copie complète affinée, valeur_train.py)
    recherche:sims:cpuct:fpu:k   recherche PUCT (recherche.py)
    juge:k:alpha                 juge v1 : petite tête sur le tronc gelé (engine_juge.py)
    sfsel:k:profondeur           Stockfish choisit dans le top-k du modèle (mesure du
                                 plafond, engine_sfselect.py ; Stockfish ne joue pas
                                 dans le bot, c'est un instrument de mesure)
Ces deux derniers jouent partie par partie (pas de mise en lot).

Exemples (depuis la racine du dépôt) :
    python -m juge_recherche.duel_lot --a valeur:5:0.25 --b base --games 2000 --out results/valeur.json
    python -m juge_recherche.duel_lot --a recherche:64:0.1:0.0:5 --b valeur:5:0.25 --games 1000 --out results/recherche.json
"""
import argparse, json, os, random, time
from collections import Counter
import numpy as np
import torch
import chess
from model import ChessGPT, ModelConfig
from juge_recherche.valeur_train import Valeur
from juge_recherche.recherche import Recherche, chercher, tronc
from elo_match import tirer_ouverture, elo_with_error

CACHE = {}
MOTEURS = []          # moteurs coup par coup, fermés en fin de programme


def politique(path, dev):
    if ("p", path) not in CACHE:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        m = ChessGPT(ModelConfig(**ck["model_config"])); m.load_state_dict(ck["model"])
        CACHE[("p", path)] = m.to(dev).eval()
    return CACHE[("p", path)]


def valeur(path, dev):
    if ("v", path) not in CACHE:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        m = Valeur(ChessGPT(ModelConfig(**ck["model_config"]))); m.load_state_dict(ck["modele"])
        CACHE[("v", path)] = m.to(dev).eval()
    return CACHE[("v", path)]


def lot_dernier(seqs, pad, dev):
    L = max(len(s) for s in seqs)
    x = torch.full((len(seqs), L), pad, dtype=torch.long)
    for i, s in enumerate(seqs):
        x[i, :len(s)] = torch.tensor(s)
    fin = torch.tensor([len(s) - 1 for s in seqs], device=dev)
    return x.to(dev), fin, torch.arange(len(seqs), device=dev)


class Joueur:
    def __init__(self, spec, ckpt, vpath, stoi, pad, dev, ctx=256, vocab="data/vocab.json",
                 juge="checkpoints/juge.pt"):
        self.spec, self.stoi, self.pad, self.dev, self.ctx = spec, stoi, pad, dev, ctx
        self.type, *args = spec.split(":")
        self.moteur = None
        if self.type == "juge":
            from juge_recherche.engine_juge import JugeEngine
            self.moteur = JugeEngine(ckpt, vocab, dev, k=int(args[0]), alpha=float(args[1]), juge_path=juge)
            return
        if self.type == "sfsel":
            from juge_recherche.engine_sfselect import SFSelectEngine
            self.moteur = SFSelectEngine(ckpt, vocab, dev, k=int(args[0]), depth=int(args[1]))
            MOTEURS.append(self.moteur)
            return
        self.pol = politique(ckpt, dev)
        if self.type != "base":
            self.val = valeur(vpath, dev)
        if self.type == "valeur":
            self.k, self.alpha = int(args[0]), float(args[1])
        if self.type == "recherche":
            self.sims, self.cpuct, self.fpu, self.k = int(args[0]), float(args[1]), float(args[2]), int(args[3])

    @torch.no_grad()
    def politique_lot(self, ids):
        out = []
        for i in range(0, len(ids), 256):
            x, fin, r = lot_dernier([s[-self.ctx:] for s in ids[i:i + 256]], self.pad, self.dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out.append(self.pol.lm_head(tronc(self.pol, x)[r, fin]).float().cpu())
        return torch.cat(out)

    @torch.no_grad()
    def jouer(self, boards, ids, historiques):
        if self.moteur is not None:                    # coup par coup
            coups = []
            for b, h in zip(boards, historiques):
                self.moteur.set_position(b, h)
                coups.append(self.moteur.best_move(b))
            return coups
        if self.type == "recherche":
            coups = []
            for i in range(0, len(boards), 128):
                R = [Recherche(b, s, self.stoi, k=self.k, cpuct=self.cpuct, fpu=self.fpu, ctx=self.ctx)
                     for b, s in zip(boards[i:i + 128], ids[i:i + 128])]
                chercher(R, self.pol, self.val, self.sims, self.pad, self.dev, lot=8)
                coups += [r.choix() for r in R]
            return coups
        lg = self.politique_lot(ids)
        legaux = [list(b.legal_moves) for b in boards]
        if self.type == "base":
            return [max(L, key=lambda m: lg[i, self.stoi[m.uci()]].item()) for i, L in enumerate(legaux)]
        # valeur : top-k de la politique, puis une note par candidat
        cands, lps = [], []
        for i, L in enumerate(legaux):
            toks = [self.stoi[m.uci()] for m in L]
            lp = torch.log_softmax(lg[i], -1)[toks]
            pv, ix = torch.topk(lp, min(self.k, len(L)))
            cands.append([L[j] for j in ix.tolist()]); lps.append(pv.tolist())
        seqs = [(s + [self.stoi[m.uci()]])[-self.ctx:] for s, cs in zip(ids, cands) for m in cs]
        notes = []
        for i in range(0, len(seqs), 1024):
            x, fin, r = lot_dernier(seqs[i:i + 1024], self.pad, self.dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                notes += self.val(x)[r, fin].float().cpu().tolist()
        coups, j = [], 0
        for cs, lp in zip(cands, lps):
            n = len(cs); sc = [notes[j + q] + self.alpha * lp[q] for q in range(n)]; j += n
            coups.append(cs[max(range(n), key=sc.__getitem__)])
        return coups


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--a", required=True); p.add_argument("--b", required=True)
    p.add_argument("--ckpt-a", default="checkpoints/run2_best.pt")
    p.add_argument("--ckpt-b", default="checkpoints/run2_best.pt")
    p.add_argument("--valeur-a", default="checkpoints/valeur.pt")
    p.add_argument("--valeur-b", default="checkpoints/valeur.pt")
    p.add_argument("--juge", default="checkpoints/juge.pt", help="juge v1 (joueurs juge:k:alpha)")
    p.add_argument("--vocab", default="data/vocab.json")
    p.add_argument("--games", type=int, default=1000)
    p.add_argument("--random-plies", type=int, default=4)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--max-plies", type=int, default=400)
    p.add_argument("--label-a", default="A"); p.add_argument("--label-b", default="B")
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda:0")
    a = p.parse_args()
    dev = a.device
    v = json.load(open(a.vocab)); stoi, bos, pad = v["stoi"], v["bos_id"], v["pad_id"]
    JA = Joueur(a.a, a.ckpt_a, a.valeur_a, stoi, pad, dev, vocab=a.vocab, juge=a.juge)
    JB = Joueur(a.b, a.ckpt_b, a.valeur_b, stoi, pad, dev, vocab=a.vocab, juge=a.juge)
    n = a.games + (a.games % 2)
    rng = random.Random(a.seed)
    parties = []
    for i in range(n):
        if i % 2 == 0:
            ouv = tirer_ouverture(rng, a.random_plies)
        b = chess.Board()
        for u in ouv:
            b.push_uci(u)
        parties.append({"b": b, "h": list(ouv), "a_blancs": i % 2 == 0, "res": None, "fin": None})
    print(f"[protocole] {n} parties en parallele, ouvertures appariees ({a.random_plies} demi-coups "
          f"aleatoires, graine {a.seed}), couleurs alternees | A = {a.label_a} ({a.a}), B = {a.label_b} ({a.b})",
          flush=True)
    t0 = time.time(); tour = 0

    def terminer(g):
        if len(g["h"]) >= a.max_plies:
            g["res"], g["fin"] = 0.5, "limite_de_coups"; return
        o = g["b"].outcome(claim_draw=True)
        if o is None:
            return
        g["fin"] = o.termination.name.lower()
        g["res"] = 0.5 if o.winner is None else (1.0 if (o.winner == chess.WHITE) == g["a_blancs"] else 0.0)

    for g in parties:
        terminer(g)
    while True:
        actives = [g for g in parties if g["res"] is None]
        if not actives:
            break
        for J, a_joue in ((JA, True), (JB, False)):
            gs = [g for g in actives if g["res"] is None and ((g["b"].turn == chess.WHITE) == g["a_blancs"]) == a_joue]
            if not gs:
                continue
            ids = [[bos] + [stoi[u] for u in g["h"]] for g in gs]
            coups = J.jouer([g["b"] for g in gs], ids, [list(g["h"]) for g in gs])
            for g, m in zip(gs, coups):
                g["b"].push(m); g["h"].append(m.uci()); terminer(g)
        tour += 1
        if tour % 10 == 0:
            fin = [g for g in parties if g["res"] is not None]
            s = sum(g["res"] for g in fin)
            print(f"  tour {tour} : {len(fin)}/{n} terminees, score A {s/max(1,len(fin)):.3f} "
                  f"| {(time.time()-t0)/60:.1f} min", flush=True)

    w = sum(g["res"] == 1.0 for g in parties); d = sum(g["res"] == 0.5 for g in parties)
    l = sum(g["res"] == 0.0 for g in parties)
    r = elo_with_error(w, d, l)
    if isinstance(r, dict):
        elo = r.get("elo_diff", r.get("elo")); bas, haut = r.get("elo_diff_ic95_bas"), r.get("elo_diff_ic95_haut")
    else:
        elo, bas, haut = (list(r) + [None] * 3)[:3]
    rapport = {"a": {"label": a.label_a, "spec": a.a, "ckpt": a.ckpt_a, "valeur": a.valeur_a},
               "b": {"label": a.label_b, "spec": a.b, "ckpt": a.ckpt_b, "valeur": a.valeur_b},
               "protocole": {"parties": n, "ouvertures_appariees": True, "demi_coups_aleatoires": a.random_plies,
                             "graine": a.seed, "en_lot": True},
               "victoires": w, "nulles": d, "defaites": l, "score": (w + 0.5 * d) / n,
               "elo_diff": elo, "elo_ic95_bas": bas, "elo_ic95_haut": haut,
               "terminaisons": dict(Counter(g["fin"] for g in parties)),
               "demi_coups_moyens": float(np.mean([len(g["h"]) for g in parties])),
               "duree_s": round(time.time() - t0, 1)}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(rapport, open(a.out, "w"), indent=2, ensure_ascii=False)
    print(f"\n=== {a.label_a} contre {a.label_b} ===")
    print(f"{n} parties : {w} victoires, {d} nulles, {l} défaites")
    fmt = lambda x: "infini" if x is None else f"{x:+.1f}"     # score de 0 ou 1 : écart non borné
    print(f"écart d'Elo  : {fmt(elo)}  [{fmt(bas)} ; {fmt(haut)}]  | {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        for m in MOTEURS:           # Stockfish : sinon le programme ne se termine pas
            m.close()
