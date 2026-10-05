"""Graphes CUDA pour l'évaluation des feuilles de la recherche.

Une passe de 142 M paramètres sur un lot de 8 à 32 séquences est limitée par le
LANCEMENT des opérations depuis Python (des milliers de petits noyaux par
passe), pas par le calcul : ~38 ms par passe mesurées sur la 3060 pour ~7 ms
de calcul. Un graphe CUDA enregistre la passe une fois (politique : tronc +
lm_head au dernier jeton ; juge : note au dernier jeton) et la rejoue d'un seul
appel.

Formes fixes obligatoires : chaque lot est complété à B séquences (lignes de
<pad>) et à une longueur palier L (bourrage à droite, sans effet dans un modèle
causal). Un graphe par couple (B, L), capturés à la demande et gardés.
"""
import torch

PALIERS_L = (48, 64, 96, 128, 160, 192, 224, 256)


class _Graphe:
    def __init__(self, politique, valeur, B, L, pad, dev, pool):
        from juge_recherche.recherche import tronc
        self.x = torch.full((B, L), pad, dtype=torch.long, device=dev)
        self.fin = torch.zeros(B, dtype=torch.long, device=dev)
        self.r = torch.arange(B, device=dev)

        def passe():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg = politique.lm_head(tronc(politique, self.x)[self.r, self.fin]).float()
                v = torch.sigmoid(valeur(self.x)[self.r, self.fin].float())
            return lg, v

        s = torch.cuda.Stream(device=dev)
        s.wait_stream(torch.cuda.current_stream(dev))
        with torch.cuda.stream(s), torch.no_grad():
            for _ in range(2):                         # échauffement hors capture
                passe()
        torch.cuda.current_stream(dev).wait_stream(s)
        self.g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.g, pool=pool), torch.no_grad():
            self.lg, self.v = passe()


class Evaluateur:
    """Remplace recherche.evaluer : mêmes entrées, mêmes sorties."""

    def __init__(self, politique, valeur, pad, dev, tailles_lot=(8, 32)):
        self.pol, self.val, self.pad, self.dev = politique, valeur, pad, dev
        self.tailles = tuple(sorted(tailles_lot))
        self.graphes = {}
        self.pool = torch.cuda.graph_pool_handle()

    def _graphe(self, B, L):
        if (B, L) not in self.graphes:
            self.graphes[(B, L)] = _Graphe(self.pol, self.val, B, L, self.pad, self.dev, self.pool)
        return self.graphes[(B, L)]

    @torch.no_grad()
    def __call__(self, seqs):
        n = len(seqs)
        B = next((b for b in self.tailles if b >= n), None)
        Lm = max(len(s) for s in seqs)
        L = next((l for l in PALIERS_L if l >= Lm), None)
        if B is None or L is None:                     # hors formes prévues : calcul normal
            from juge_recherche.recherche import evaluer
            return evaluer(self.pol, self.val, seqs, self.pad, self.dev)
        g = self._graphe(B, L)
        x = torch.full((B, L), self.pad, dtype=torch.long)
        fin = torch.zeros(B, dtype=torch.long)
        for i, s in enumerate(seqs):
            x[i, :len(s)] = torch.tensor(s)
            fin[i] = len(s) - 1
        g.x.copy_(x.to(self.dev, non_blocking=True))
        g.fin.copy_(fin.to(self.dev, non_blocking=True))
        g.g.replay()
        return g.lg[:n].cpu(), g.v[:n].cpu().tolist()
