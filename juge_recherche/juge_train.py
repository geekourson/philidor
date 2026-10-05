"""Étape 1 : entraînement du juge, une tête de valeur sur le tronc gelé.

Entrée : l'état caché de run2 APRÈS un coup candidat (bloc 15 + sortie,
1 536 dimensions). Sortie : la probabilité de gain du joueur qui vient de jouer
ce coup, cible `1/(1+10^(-cp/400))` à partir de l'évaluation Stockfish.

Deux pertes :
  - une régression (BCE) sur la probabilité de gain, pour que la note ait un
    sens absolu ;
  - un classement (entropie croisée sur les candidats d'une même position, cible
    = le meilleur selon Stockfish). C'est la leçon de la tête Q : une cible
    absolue seule dépense la capacité à prédire le niveau de la position,
    commun à tous les candidats, au lieu de les départager. Or le juge ne sert
    qu'à départager.
"""
import argparse, collections, json, math, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F


class Juge(nn.Module):
    def __init__(self, d, h=512):
        super().__init__()
        self.register_buffer("mu", torch.zeros(d))
        self.register_buffer("sd", torch.ones(d))
        self.net = nn.Sequential(nn.Linear(d, h), nn.GELU(), nn.Dropout(0.1),
                                 nn.Linear(h, h), nn.GELU(), nn.Linear(h, 1))

    def forward(self, x):
        return self.net((x - self.mu) / self.sd).squeeze(-1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", default="data/juge/train")
    p.add_argument("--out", default="checkpoints/juge.pt")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--lot", type=int, default=2048)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lam", type=float, default=1.0, help="poids de la perte de classement")
    p.add_argument("--val-frac", type=float, default=0.05)
    a = p.parse_args()
    dev = a.device
    t0 = time.time()

    X = np.load(a.train + ".X.npy", mmap_mode="r")
    meta = json.load(open(a.train + ".meta.json"))
    groupes = collections.defaultdict(list)
    for i, m in enumerate(meta):
        groupes[m["r"]].append(i)
    racines = sorted(groupes)
    rng = np.random.default_rng(7); rng.shuffle(racines)
    nv = int(len(racines) * a.val_frac)
    val_r, tr_r = racines[:nv], racines[nv:]

    def tenseurs(rs):
        idx = np.full((len(rs), 3), -1, dtype=np.int64)
        for j, r in enumerate(rs):
            g = groupes[r][:3]; idx[j, :len(g)] = g
        cp = np.array([[meta[i]["cp"] if i >= 0 else 0.0 for i in row] for row in idx],
                      dtype=np.float32)
        return torch.tensor(idx), torch.tensor(cp)

    Xg = torch.tensor(np.asarray(X), dtype=torch.float16, device=dev)
    itr, cptr = tenseurs(tr_r); iva, cpva = tenseurs(val_r)
    print(f"[juge] {len(meta):,} candidats, {len(tr_r):,} positions d'entrainement, "
          f"{len(val_r):,} de validation ({time.time()-t0:.0f}s)", flush=True)

    juge = Juge(X.shape[1]).to(dev)
    echant = Xg[torch.randint(0, len(Xg), (200_000,), device=dev)].float()
    juge.mu.copy_(echant.mean(0)); juge.sd.copy_(echant.std(0) + 1e-3)
    opt = torch.optim.AdamW(juge.parameters(), lr=a.lr, weight_decay=1e-2)
    etapes = a.epochs * math.ceil(len(tr_r) / a.lot)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=etapes)

    def passe(idx, cp, entrainer):
        juge.train(entrainer)
        masque = idx >= 0
        x = Xg[idx.clamp(min=0).to(dev)].float()           # (B, 3, d)
        s = juge(x)                                        # (B, 3)
        cpd = cp.to(dev); m = masque.to(dev)
        cible = torch.sigmoid(cpd * math.log(10) / 400)
        bce = (F.binary_cross_entropy_with_logits(s, cible, reduction="none") * m).sum() / m.sum()
        s_m = s.masked_fill(~m, -1e9)
        meilleur = cpd.masked_fill(~m, -1e9).argmax(1)
        plusieurs = m.sum(1) >= 2
        ce = F.cross_entropy(s_m[plusieurs], meilleur[plusieurs]) if plusieurs.any() else s.sum() * 0
        choix = s_m.argmax(1)
        juste = (cpd.gather(1, choix[:, None]).squeeze(1) >=
                 cpd.masked_fill(~m, -1e9).max(1).values - 1e-6)[plusieurs]
        return bce, ce, juste

    meilleur_val = -1.0
    for ep in range(1, a.epochs + 1):
        perm = torch.randperm(len(itr))
        for k in range(0, len(perm), a.lot):
            b = perm[k:k + a.lot]
            bce, ce, _ = passe(itr[b], cptr[b], True)
            loss = bce + a.lam * ce
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        with torch.no_grad():
            tot = []; bces = []; ces = []
            for k in range(0, len(iva), 8192):
                bce, ce, j = passe(iva[k:k + 8192], cpva[k:k + 8192], False)
                tot.append(j.float()); bces.append(float(bce)); ces.append(float(ce))
            acc = float(torch.cat(tot).mean())
        marque = ""
        if acc > meilleur_val:
            meilleur_val = acc
            torch.save({"etat": juge.state_dict(), "d": X.shape[1], "val_choix": acc}, a.out)
            marque = "  <- meilleur"
        print(f"  epoque {ep:2d} | val BCE {np.mean(bces):.4f} | val classement {np.mean(ces):.4f} "
              f"| choisit le meilleur candidat {acc:.2%}{marque} ({time.time()-t0:.0f}s)", flush=True)
    print(f"\n-> {a.out} | meilleur choix en validation {meilleur_val:.2%}")


if __name__ == "__main__":
    main()
