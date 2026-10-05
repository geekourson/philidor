"""Étape B, étiquetage : le sélecteur « politique + valeur » sur des parties humaines.

Itération experte, version minimale : on fait tourner le sélecteur sur des
millions de positions, puis on affine la politique pour qu'elle joue
directement le coup reclassé (iteration_train.py). Le gain du sélecteur passe
alors dans la passe unique.

Pour chaque partie (corpus d'entraînement de run2, parties de validation
sautées), à chaque position à partir du demi-coup `--ply-min` :
  - une passe de la politique sur toute la partie (causale : toutes les
    distributions d'un coup), masque de légalité, top-k ;
  - une passe du réseau de valeur par candidat, sur « préfixe + candidat »,
    toutes positions et parties mêlées dans des lots triés par longueur
    (bourrage à droite, lecture au dernier token réel).
On garde les notes brutes (logit de valeur, log-probabilité de la politique)
pour choisir la forme de la cible à l'entraînement sans réétiqueter.

Sorties (--out) : parties.u16 + parties_off.i64 (séquences <bos>+coups),
pos.i32 (partie, demi-coup), cand.u16 (k), lv.f16 (k), lp.f16 (k).
"""
import argparse, json, os, time
import numpy as np
import torch
import chess
from model import ChessGPT, ModelConfig
from juge_recherche.valeur_train import Valeur
from juge_recherche.recherche import tronc


def rot(x, cos, sin):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


def forward_arbre(val, x, P, M):
    """Réseau de valeur sur des parties « emballées » : la partie, puis tous les
    candidats de toutes ses positions à la suite. x (B,L) jetons, P (B,L)
    positions RoPE (un candidat de la position t prend la position t+1, celle
    du vrai coup qu'il remplace), M (B,L,L) masque booléen : un vrai coup voit
    les vrais coups qui le précèdent, un candidat voit le préfixe de sa position
    et lui-même, rien d'autre. Mêmes opérations que model.py, dans le même ordre."""
    g = val.gpt
    h = g.tok_emb(x)
    cos, sin = g.rope_cos[P][:, None], g.rope_sin[P][:, None]
    m = M[:, None]
    for blk in g.blocks:
        at = blk.attn
        z = blk.norm_attn(h)
        B_, T_, C_ = z.shape
        q, k, v = at.qkv(z).split(C_, dim=2)
        q = q.view(B_, T_, at.n_head, at.head_dim).transpose(1, 2)
        k = k.view(B_, T_, at.n_head, at.head_dim).transpose(1, 2)
        v = v.view(B_, T_, at.n_head, at.head_dim).transpose(1, 2)
        y = torch.nn.functional.scaled_dot_product_attention(rot(q, cos, sin), rot(k, cos, sin), v, attn_mask=m)
        h = h + at.proj(y.transpose(1, 2).contiguous().view(B_, T_, C_))
        h = h + blk.mlp(blk.norm_mlp(h))
    return val.tete(g.norm_final(h)).squeeze(-1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--games", default="data/games_uci.txt")
    p.add_argument("--val", default="data/val_games.txt")
    p.add_argument("--debut", type=int, default=0, help="ligne de départ dans --games")
    p.add_argument("--n-parties", type=int, default=100_000)
    p.add_argument("--ply-min", type=int, default=8)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--valeur", default="checkpoints/valeur.pt")
    p.add_argument("--politique", default="checkpoints/run2_best.pt")
    p.add_argument("--out", default="data/iteration/b1")
    p.add_argument("--lot-parties", type=int, default=64)
    p.add_argument("--lot-seqs", type=int, default=1024)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--sequences", action="store_true", help="ancien calcul, une séquence par candidat (contrôle)")
    p.add_argument("--jetons-lot", type=int, default=24000)
    a = p.parse_args()
    dev = a.device

    v = json.load(open("data/vocab.json")); stoi, bos, pad = v["stoi"], v["bos_id"], v["pad_id"]
    ck = torch.load(a.politique, map_location="cpu", weights_only=False)
    pol = ChessGPT(ModelConfig(**ck["model_config"])); pol.load_state_dict(ck["model"]); pol = pol.to(dev).eval()
    cv = torch.load(a.valeur, map_location="cpu", weights_only=False)
    val = Valeur(ChessGPT(ModelConfig(**cv["model_config"]))); val.load_state_dict(cv["modele"])
    val = val.to(dev).eval()
    ctx = pol.cfg.block_size

    parties_val = {l.strip() for l in open(a.val)}
    os.makedirs(a.out, exist_ok=True)
    G, GO, POS, CAND, LV, LP = [], [0], [], [], [], []
    t0 = time.time(); n_lu = n_val = 0

    def lire():
        nonlocal n_lu, n_val
        with open(a.games) as f:
            for i, l in enumerate(f):
                if i < a.debut:
                    continue
                s = l.strip()
                if s in parties_val:
                    n_val += 1; continue
                u = s.split()
                if len(u) < a.ply_min + 10 or any(m not in stoi for m in u):
                    continue
                n_lu += 1
                yield u[:ctx - 1]
                if n_lu >= a.n_parties:
                    return

    @torch.no_grad()
    def traiter(parties):
        # 1) politique : une passe par partie, lots par longueur
        seqs = [[bos] + [stoi[m] for m in u] for u in parties]
        L = max(len(s) for s in seqs)
        x = torch.full((len(seqs), L), pad, dtype=torch.long)
        for i, s in enumerate(seqs):
            x[i, :len(s)] = torch.tensor(s)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg = pol.lm_head(tronc(pol, x.to(dev))).float()
        lsm = torch.log_softmax(lg, -1).cpu()
        # 2) candidats légaux
        demandes = []                    # (index partie, demi-coup, [cand toks], [logp])
        for gi, u in enumerate(parties):
            b = chess.Board()
            for t, m in enumerate(u):
                if t >= a.ply_min:
                    leg = [stoi[mm.uci()] for mm in b.legal_moves]
                    if len(leg) >= 2:
                        lpl = lsm[gi, t, leg]
                        kk = min(a.k, len(leg))
                        pv, ix = torch.topk(lpl, kk)
                        demandes.append((gi, t, [leg[j] for j in ix.tolist()], pv.tolist()))
                b.push_uci(m)
        # 3) valeur de chaque candidat
        if not a.sequences:
            notes = valeurs_emballees(seqs, demandes)
        else:
            notes = valeurs_sequences(seqs, demandes)
        j = 0
        base = len(GO) - 1
        for s in seqs:
            G.append(np.asarray(s, np.uint16)); GO.append(GO[-1] + len(s))
        for gi, t, cands, lps in demandes:
            n = len(cands)
            c5 = cands + [pad] * (a.k - n); l5 = lps + [-1e4] * (a.k - n)
            v5 = list(notes[j:j + n]) + [-1e4] * (a.k - n); j += n
            POS.append((base + gi, t)); CAND.append(c5); LP.append(l5); LV.append(v5)

    @torch.no_grad()
    def valeurs_emballees(seqs, demandes):
        par = [[] for _ in seqs]                        # par partie : (t, cands, rang du 1er candidat)
        rang = 0
        for gi, t, cands, _ in demandes:
            par[gi].append((t, cands, rang)); rang += len(cands)
        notes = np.zeros(rang, np.float32)
        paquets = []
        for gi, s in enumerate(seqs):
            if par[gi]:
                paquets.append((gi, len(s) + sum(len(c) for _, c, _ in par[gi])))
        paquets.sort(key=lambda z: z[1])
        i0 = 0
        while i0 < len(paquets):
            i1 = i0
            while i1 < len(paquets) and paquets[i1][1] * (i1 - i0 + 1) <= a.jetons_lot:
                i1 += 1
            i1 = max(i1, i0 + 1)
            lot = paquets[i0:i1]; Lm = lot[-1][1]
            x = np.full((len(lot), Lm), pad, np.int64); P = np.zeros((len(lot), Lm), np.int64)
            M = np.zeros((len(lot), Lm, Lm), bool)
            lire = []                                   # (ligne, colonne, rang de la note)
            for r, (gi, Lg) in enumerate(lot):
                s = seqs[gi]; n = len(s)
                x[r, :n] = s; P[r, :n] = np.arange(n)
                M[r, :n, :n] = np.tril(np.ones((n, n), bool))
                q = n
                for t, cands, rg in par[gi]:
                    for c_i, c in enumerate(cands):
                        x[r, q] = c; P[r, q] = t + 1
                        M[r, q, :t + 1] = True; M[r, q, q] = True
                        lire.append((r, q, rg + c_i)); q += 1
                for qq in range(q, Lm):                 # bourrage : ne voit que lui-même
                    M[r, qq, qq] = True
            xt, Pt, Mt = (torch.from_numpy(z).to(dev) for z in (x, P, M))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                o = forward_arbre(val, xt, Pt, Mt).float().cpu().numpy()
            for r, q, rg in lire:
                notes[rg] = o[r, q]
            i0 = i1
        return notes

    @torch.no_grad()
    def valeurs_sequences(seqs, demandes):
        items = [(di, c) for di, d in enumerate(demandes) for c in range(len(d[2]))]
        longs = [demandes[di][1] + 2 for di, _ in items]
        ordre = np.argsort(longs, kind="stable")
        notes = np.zeros(len(items), np.float32)
        for k0 in range(0, len(items), a.lot_seqs):
            bi = ordre[k0:k0 + a.lot_seqs]
            Lb = max(longs[i] for i in bi)
            xx = torch.full((len(bi), Lb), pad, dtype=torch.long)
            for r, i in enumerate(bi):
                di, c = items[i]; gi, t, cands, _ = demandes[di]
                s = seqs[gi][:t + 1] + [cands[c]]
                xx[r, :len(s)] = torch.tensor(s)
            fin = torch.tensor([longs[i] - 1 for i in bi], device=dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                o = val(xx.to(dev))[torch.arange(len(bi), device=dev), fin].float()
            notes[bi] = o.cpu().numpy()
        return notes

    lot = []
    for u in lire():
        lot.append(u)
        if len(lot) == a.lot_parties:
            traiter(lot); lot = []
            if n_lu % 5000 < a.lot_parties:
                dt = time.time() - t0
                print(f"  {n_lu:,} parties, {len(POS):,} positions ({dt/60:.1f} min, "
                      f"{n_lu/dt:.1f} parties/s)", flush=True)
    if lot:
        traiter(lot)

    np.concatenate(G).tofile(f"{a.out}/parties.u16")
    np.asarray(GO, np.int64).tofile(f"{a.out}/parties_off.i64")
    np.asarray(POS, np.int32).tofile(f"{a.out}/pos.i32")
    np.asarray(CAND, np.uint16).tofile(f"{a.out}/cand.u16")
    np.asarray(LV, np.float16).tofile(f"{a.out}/lv.f16")
    np.asarray(LP, np.float16).tofile(f"{a.out}/lp.f16")
    json.dump({"k": a.k, "n_parties": n_lu, "n_positions": len(POS), "ply_min": a.ply_min,
               "debut": a.debut, "val_sautees": n_val, "valeur": a.valeur, "politique": a.politique},
              open(f"{a.out}/info.json", "w"))
    print(f"[b] {n_lu:,} parties, {len(POS):,} positions, {n_val:,} parties de validation sautees "
          f"| {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
