"""Étape A : une petite recherche arborescente, la politique guide, la valeur note.

Recherche PUCT à la AlphaZero, tout en passes du transformeur, aucune
évaluation écrite à la main :
  - la politique (run2, inchangée) donne les probabilités a priori des coups ;
  - le réseau de valeur (valeur_train.py) note chaque position atteinte, du
    point de vue du joueur qui vient de jouer, en probabilité de gain
    sigmoid(logit).

Une simulation descend l'arbre en maximisant Q + cpuct · P · sqrt(N_parent) /
(1 + N_enfant), évalue la feuille (UNE passe de chaque réseau sur la séquence
« historique + chemin »), la développe en ses k coups les plus probables, et
remonte sa valeur en alternant le point de vue à chaque demi-coup. Le coup joué
est le plus visité.

Première visite (FPU) : un enfant jamais visité reçoit la valeur de son parent,
vue de son camp, moins `fpu`. Les fins de partie (mat, pat, matériel
insuffisant, triple répétition, 50 coups) sont notées exactement et jamais
développées.

MISE EN LOT. Les feuilles de nombreuses recherches (plusieurs positions à la
fois, et `lot` feuilles par recherche grâce à la perte virtuelle) sont évaluées
ensemble. Séquences de longueurs différentes : bourrage à DROITE, sans effet
dans un modèle causal (contrôlé dans valeur_train.py), et lecture au dernier
token réel de chaque séquence.
"""
import math
import chess
import torch
import torch.nn.functional as F


class Noeud:
    __slots__ = ("coup", "tok", "p", "n", "w", "enfants", "terminal", "attente")

    def __init__(self, coup, tok, p):
        self.coup, self.tok, self.p = coup, tok, p
        self.n = 0            # visites (perte virtuelle comprise pendant l'attente)
        self.w = 0.0          # somme des valeurs, point de vue du joueur qui a joué `coup`
        self.enfants = None   # None = pas encore développé
        self.terminal = None  # None = pas encore testé, False = partie en cours, sinon valeur exacte
        self.attente = False


def valeur_terminale(b):
    """Valeur exacte pour le joueur qui vient de jouer, ou False si la partie continue."""
    if b.is_checkmate():
        return 1.0
    if (b.is_stalemate() or b.is_insufficient_material() or b.halfmove_clock >= 100
            or b.is_repetition(3)):
        return 0.5
    return False


class Recherche:
    def __init__(self, board, ids, stoi, k=5, cpuct=2.0, fpu=0.1, ctx=256):
        self.board0, self.ids0, self.stoi = board, list(ids), stoi
        self.k, self.cpuct, self.fpu, self.ctx = k, cpuct, fpu, ctx
        self.racine = Noeud(None, None, 1.0)
        self.racine.terminal = False
        self.faites = 0

    def selectionner(self):
        """-> demande d'évaluation (chemin, plateau, ids), None si fin de partie
        (déjà remontée), ou "collision" si la feuille attend déjà son évaluation."""
        nd, b, ids, chemin = self.racine, self.board0.copy(), list(self.ids0), [self.racine]
        while nd.enfants is not None:
            sq = math.sqrt(max(nd.n, 1))
            q_fpu = (1.0 - nd.w / nd.n if nd.n else 0.5) - self.fpu
            meilleur, sm = None, -1e18
            for c in nd.enfants:
                q = c.w / c.n if c.n else q_fpu
                s = q + self.cpuct * c.p * sq / (1 + c.n)
                if s > sm:
                    meilleur, sm = c, s
            nd = meilleur
            b.push(nd.coup); ids.append(nd.tok); chemin.append(nd)
            if nd.terminal is None:
                nd.terminal = valeur_terminale(b)
            if nd.terminal is not False:
                break
        if nd.terminal is not False:                      # fin de partie : valeur exacte
            v = nd.terminal
            for x in reversed(chemin):
                x.n += 1; x.w += v; v = 1.0 - v
            return None
        if nd.attente:
            return "collision"
        nd.attente = True
        for x in chemin:                                   # perte virtuelle
            x.n += 1
        return chemin, b, ids[-self.ctx:]

    def developper(self, demande, logits, v):
        chemin, b, _ = demande
        nd = chemin[-1]
        legaux = list(b.legal_moves)
        toks = [self.stoi[m.uci()] for m in legaux]
        p = torch.softmax(logits[toks], 0)
        kk = min(self.k, len(legaux))
        pv, ix = torch.topk(p, kk)
        pv = (pv / pv.sum()).tolist()
        nd.enfants = [Noeud(legaux[i], toks[i], pp) for i, pp in zip(ix.tolist(), pv)]
        nd.attente = False
        for x in reversed(chemin):                         # n déjà compté (perte virtuelle)
            x.w += v; v = 1.0 - v

    def choix(self):
        cs = self.racine.enfants
        return max(cs, key=lambda c: (c.n, c.w / c.n if c.n else -1)).coup


def tronc(gpt, x):
    T = x.shape[1]
    h = gpt.tok_emb(x); cos, sin = gpt.rope_cos[:T], gpt.rope_sin[:T]
    for blk in gpt.blocks:
        h = blk(h, cos, sin)
    return gpt.norm_final(h)


@torch.no_grad()
def evaluer(politique, valeur, seqs, pad, dev):
    L = max(len(s) for s in seqs)
    x = torch.full((len(seqs), L), pad, dtype=torch.long)
    for i, s in enumerate(seqs):
        x[i, :len(s)] = torch.tensor(s)
    x = x.to(dev)
    fin = torch.tensor([len(s) - 1 for s in seqs], device=dev)
    r = torch.arange(len(seqs), device=dev)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = politique.lm_head(tronc(politique, x)[r, fin]).float()
        v = torch.sigmoid(valeur(x)[r, fin].float())
    return logits.cpu(), v.cpu().tolist()


def chercher(recherches, politique, valeur, sims, pad, dev, lot=8, arret=None, evaluateur=None):
    """Fait avancer toutes les recherches de `sims` simulations, feuilles évaluées en lot.

    `arret` (facultatif) : fonction consultée entre deux lots ; si elle renvoie
    vrai, on s'arrête là (racines déjà développées) et chaque recherche joue son
    meilleur coup du moment. Sert au bot : une autre partie attend le serveur.
    `evaluateur` (facultatif) : remplace `evaluer` (graphes.Evaluateur, graphes CUDA).
    """
    while True:
        if arret is not None and all(r.racine.enfants is not None for r in recherches) and arret():
            return
        demandes = []
        for r in recherches:
            for _ in range(min(lot, sims - r.faites)):
                d = r.selectionner()
                if d == "collision":
                    break
                r.faites += 1
                if d is not None:
                    demandes.append((r, d))
        if not demandes:
            if all(r.faites >= sims for r in recherches):
                return
            continue
        seqs = [d[2] for _, d in demandes]
        logits, v = evaluateur(seqs) if evaluateur is not None else evaluer(politique, valeur, seqs, pad, dev)
        for (r, d), lg, vv in zip(demandes, logits, v):
            r.developper(d, lg, vv)
