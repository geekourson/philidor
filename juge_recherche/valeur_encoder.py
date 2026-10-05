"""Réseau de valeur, étape 1 : les parties annotées de Lichess en séquences + cibles.

Chaque partie de eval_2026-05.jsonl porte une évaluation Stockfish après chaque
coup (du point de vue des BLANCS, mats bornés à ±10000 ; signe vérifié sur 1 247
mats de fin de partie, 100 % cohérents avec le résultat). Le modèle est causal :
une seule passe sur la partie donne un état après chaque coup, donc une cible
par token, environ 70 par séquence au lieu d'une.

Convention de la cible : au token du coup i, la note est celle de la position
obtenue, vue par le joueur QUI VIENT DE JOUER ce coup (Blancs si i est pair).
C'est exactement la question du choix : « après mon coup, où en suis-je ? ».

Contamination : on exclut toute partie qui passe, entre les demi-coups 28 et 82,
par une position de mesure (5 000 du diagnostic, 4 670 du bot) ou par la
position obtenue après l'un de leurs cinq premiers candidats. Sinon le verrou
mesurerait de la mémoire.

Sorties (dans --out) : tokens.u16, cp.i16 (même alignement, -32768 = pas de
cible), offsets.i64 (n+1), split.u8 (0 entraînement, 1 validation, 2 exclue).
"""
import argparse, hashlib, json, os, time
from multiprocessing import Pool
import numpy as np
import chess

SANS = -32768
PLY_MIN, PLY_MAX = 28, 82
CTX = 256

STOI = None
INTERDITS = None


def h(epd):
    return hashlib.blake2b(epd.encode(), digest_size=8).digest()


def init(stoi, interdits):
    global STOI, INTERDITS
    STOI, INTERDITS = stoi, interdits


def traiter(args):
    n, ligne = args
    d = json.loads(ligne)
    uci, cp = d["uci"][:CTX - 1], d["cp"][:CTX - 1]
    toks = [STOI["<bos>"]]
    cibles = [SANS]
    for i, (u, c) in enumerate(zip(uci, cp)):
        if u not in STOI:
            return n, None, None, "vocab"
        toks.append(STOI[u])
        if c is None:
            cibles.append(SANS)
        else:
            c = max(-10000, min(10000, int(c)))
            cibles.append(c if i % 2 == 0 else -c)
    b = chess.Board()
    for i, u in enumerate(uci[:PLY_MAX + 1]):
        try:
            b.push_uci(u)
        except ValueError:
            return n, None, None, "illegal"
        if i + 1 >= PLY_MIN and h(b.epd()) in INTERDITS:
            return n, toks, cibles, "contamine"
    return n, toks, cibles, "ok"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", nargs="+", default=["data/eval_pgn/eval_2026-05.jsonl"],
                   help="un ou plusieurs mois ; numérotation des parties continue d'un fichier à l'autre")
    p.add_argument("--out", default="data/valeur")
    p.add_argument("--vocab", default="data/vocab.json")
    p.add_argument("--max", type=int, default=0)
    p.add_argument("--val-pct", type=float, default=1.0)
    p.add_argument("--workers", type=int, default=11)
    a = p.parse_args()

    stoi = json.load(open(a.vocab))["stoi"]
    interdits = set()
    for pos, mod in (("diag/positions.jsonl", "diag/modele.jsonl"),
                     ("diag/propres_positions.jsonl", "diag/propres_modele.jsonl")):
        cls = {}
        for l in open(mod):
            m = json.loads(l); cls[m["fen"]] = m["classement"][:5]
        for l in open(pos):
            q = json.loads(l); b = chess.Board(q["fen"])
            interdits.add(h(b.epd()))
            for u in cls.get(q["fen"], []):
                b.push_uci(u); interdits.add(h(b.epd())); b.pop()
    print(f"[valeur] {len(interdits):,} positions interdites (racines + enfants)", flush=True)

    os.makedirs(a.out, exist_ok=True)
    T, C, O, S = [], [], [0], []
    compte = {"ok": 0, "contamine": 0, "vocab": 0, "illegal": 0}
    t0 = time.time()

    def lignes():
        n = 0
        for src in a.src:
            with open(src) as f:
                for l in f:
                    if a.max and n >= a.max:
                        return
                    yield n, l
                    n += 1

    with Pool(a.workers, initializer=init, initargs=(stoi, interdits)) as pool:
        for n, toks, cib, etat in pool.imap(traiter, lignes(), chunksize=256):
            compte[etat] += 1
            if toks is None:
                continue
            T.append(np.asarray(toks, np.uint16)); C.append(np.asarray(cib, np.int16))
            O.append(O[-1] + len(toks))
            if etat == "contamine":
                S.append(2)
            else:   # validation : tirage deterministe par numero de ligne
                v = int.from_bytes(hashlib.blake2b(str(n).encode(), digest_size=4).digest(), "little")
                S.append(1 if (v % 10000) < a.val_pct * 100 else 0)
            if (n + 1) % 100_000 == 0:
                print(f"  {n+1:,} parties ({time.time()-t0:.0f}s) {compte}", flush=True)

    np.concatenate(T).tofile(os.path.join(a.out, "tokens.u16"))
    np.concatenate(C).tofile(os.path.join(a.out, "cp.i16"))
    np.asarray(O, np.int64).tofile(os.path.join(a.out, "offsets.i64"))
    S = np.asarray(S, np.uint8); S.tofile(os.path.join(a.out, "split.u8"))
    cib = np.concatenate(C)
    print(f"[valeur] {compte} | entrainement {int((S==0).sum()):,}, validation {int((S==1).sum()):,}, "
          f"exclues {int((S==2).sum()):,} | {O[-1]:,} tokens, {int((cib!=SANS).sum()):,} cibles "
          f"| {(time.time()-t0)/60:.1f} min", flush=True)
    json.dump({"compte": compte, "tokens": O[-1], "cibles": int((cib != SANS).sum()),
               "ply_min": PLY_MIN, "ply_max": PLY_MAX, "val_pct": a.val_pct},
              open(os.path.join(a.out, "info.json"), "w"))


if __name__ == "__main__":
    main()
