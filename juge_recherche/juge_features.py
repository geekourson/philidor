"""Étape 1 : états cachés de run2 APRÈS chaque coup candidat, tronc gelé.

Le juge note la position obtenue après un coup, pas le coup lui-même. La tête Q
gelée essayée plus tôt lisait l'état AVANT le coup et devait deviner les
conséquences de trois coups à partir d'un seul vecteur : +0,8 point seulement.
Ici chaque candidat a sa propre passe avant, sur une séquence qui se termine par
ce coup ; c'est dans cet état que la mesure 3 du diagnostic lit « pièce en
prise » à AUC 0,992.

Deux couches : le bloc 15 (où les sondes culminaient) et la sortie de
`norm_final`, concaténées, soit 1 536 dimensions.

MISE EN LOT SANS BOURRAGE. Le modèle n'a pas de masque d'attention pour le
bourrage : la version du diagnostic qui bourrait à gauche par <bos> ne donnait
le bon coup que 70 fois sur 200. On regroupe donc les séquences par LONGUEUR
EXACTE, comme le faisait déjà q_candidats.py, et chaque lot est rectangulaire.
Contrôle intégré : les 64 premières séquences sont recalculées une par une et
comparées au calcul par lot.
"""
import argparse, collections, json, os, time
import numpy as np
import torch
from model import ChessGPT, ModelConfig


def charger_items(chemin, source, k=None, exclure=None):
    """(racine, prefixe, candidat, cp | None) pour chaque candidat."""
    items = []
    if source == "qlabels":
        for i, l in enumerate(open(chemin)):
            d = json.loads(l)
            if exclure and " ".join(d["prefixe"]) in exclure:
                continue
            for u, cp in d["q"].items():
                items.append((i, d["prefixe"], u, float(cp)))
    else:                                     # positions d'evaluation : top-k
        P = {json.loads(l)["fen"]: json.loads(l) for l in open(chemin[0])}
        for i, l in enumerate(open(chemin[1])):
            m = json.loads(l)
            if m["fen"] not in P:
                continue
            for u in m["classement"][:k]:
                items.append((i, P[m["fen"]]["prefixe"], u, None))
    return items


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/run2_best.pt")
    p.add_argument("--vocab", default="data/vocab.json")
    p.add_argument("--source", choices=["qlabels", "eval"], required=True)
    p.add_argument("--qlabels", default="data/q/q_labels.jsonl")
    p.add_argument("--positions", default="diag/positions.jsonl")
    p.add_argument("--modele", default="diag/modele.jsonl")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--exclure", nargs="*", default=[],
                   help="fichiers de positions dont les prefixes sont exclus")
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--lot", type=int, default=512)
    a = p.parse_args()

    v = json.load(open(a.vocab)); stoi, bos = v["stoi"], v["bos_id"]
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = ModelConfig(**ck["model_config"]); model = ChessGPT(cfg)
    model.load_state_dict(ck["model"]); model = model.to(a.device).eval()
    etat = {}
    model.blocks[14].register_forward_hook(lambda m, i, o: etat.__setitem__("b15", o))
    model.norm_final.register_forward_hook(lambda m, i, o: etat.__setitem__("fin", o))

    exclure = set()
    for f in a.exclure:
        for l in open(f):
            exclure.add(" ".join(json.loads(l)["prefixe"]))
    t0 = time.time()
    if a.source == "qlabels":
        items = charger_items(a.qlabels, "qlabels", exclure=exclure)
    else:
        items = charger_items((a.positions, a.modele), "eval", k=a.k)
    print(f"[juge] {len(items):,} candidats | {len(exclure):,} racines exclues "
          f"({time.time()-t0:.0f}s)", flush=True)

    seqs = []
    for (_, pref, u, _) in items:
        ids = [bos] + [stoi[m] for m in pref if m in stoi] + [stoi[u]]
        seqs.append(ids[-cfg.block_size:])
    par_long = collections.defaultdict(list)
    for i, s in enumerate(seqs):
        par_long[len(s)].append(i)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    X = np.lib.format.open_memmap(a.out + ".X.npy", mode="w+",
                                  dtype=np.float16, shape=(len(items), 2 * cfg.n_embd))
    fait = 0
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for L, idx in sorted(par_long.items()):
            for k0 in range(0, len(idx), a.lot):
                b = idx[k0:k0 + a.lot]
                x = torch.tensor([seqs[i] for i in b], dtype=torch.long, device=a.device)
                model(x)
                h = torch.cat([etat["b15"][:, -1, :], etat["fin"][:, -1, :]], -1)
                X[b] = h.float().cpu().numpy().astype(np.float16)
                fait += len(b)
            if fait and (fait // 200_000) != ((fait - len(idx)) // 200_000):
                print(f"  {fait:,}/{len(items):,} ({time.time()-t0:.0f}s)", flush=True)
    X.flush()

    # controle : lot contre une par une, sur 64 sequences de longueurs variees
    ecart = 0.0
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i in range(0, len(items), max(1, len(items) // 64))[:64]:
            model(torch.tensor([seqs[i]], dtype=torch.long, device=a.device))
            h = torch.cat([etat["b15"][0, -1], etat["fin"][0, -1]]).float().cpu().numpy()
            ecart = max(ecart, float(np.abs(h - X[i].astype(np.float32)).max()))
    print(f"[controle] ecart max lot / une par une : {ecart:.4f}", flush=True)

    json.dump([{"r": r, "u": u, "cp": cp} for (r, _, u, cp) in items],
              open(a.out + ".meta.json", "w"))
    print(f"{len(items):,} candidats -> {a.out}.X.npy "
          f"({X.nbytes/2**30:.2f} Go) en {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
