"""Récupération des évaluations Stockfish déjà présentes dans les dumps Lichess.

`prepare_data.py` supprime les commentaires PGN ligne 77 avec `COMMENT_RE`,
depuis le premier jour du projet. Or ces commentaires contiennent les
annotations `[%eval]` des parties analysées par leurs joueurs :

    1. e4 { [%eval 0.18] [%clk 0:01:00] } 1... e5 { [%eval 0.22] }

14,7 % des parties gardées par les filtres sont annotées, ~70 évaluations
chacune. Sur les 4 mois disponibles : environ 454 millions d'étiquettes déjà
sur le disque, soit ~1052 heures de Stockfish économisées.

**Ce sont des étiquettes Q, pas seulement V.** L'annotation donne l'évaluation
APRÈS le coup joué, donc pour la position d'avant :

    Q(position, coup joué) = -eval(après le coup)

Le signe s'inverse parce que Lichess note toujours du point de vue des Blancs,
alors qu'on veut le point de vue de celui qui joue. Et comme les humains
jouent de mauvais coups, on obtient exactement ce que le diagnostic réclamait :
des GAFFES notées, sur la distribution réelle des erreurs humaines.

Filtres identiques à `prepare_data.py`, pour rester comparable au corpus
existant. Une partie n'est gardée que si elle passe les filtres ET porte des
`%eval`.

Sortie : un jsonl par mois, une ligne par partie :
    {"uci": [...], "cp": [...], "res": "1-0", "elo": [2010, 1985]}
`cp[i]` est l'évaluation APRÈS le coup `uci[i]`, du point de vue des Blancs,
en centipions. Les mats sont bornés à ±10000. `null` quand le coup n'est pas
annoté (les annotations s'arrêtent souvent avant la fin).

Usage :
    python extraire_eval.py --mois 2026-04 --workers 3
"""

import argparse
import glob
import io
import json
import multiprocessing as mp
import os
import re
import time

import chess
import chess.pgn  # noqa: F401  (garantit que python-chess est bien installé)
import zstandard

import prepare_data as pd

EVAL_RE = re.compile(r"\[%eval\s+(#?-?\d+(?:\.\d+)?)\]")
# On parcourt le movetext dans l'ordre, en distinguant commentaires et jetons.
#
# PIEGE, corrige apres coup : la premiere version appliquait MOVENUM_RE
# (`\d+\.+`) au movetext AVANT d'extraire les evaluations. Or ce motif frappe
# aussi a l'interieur des commentaires : `[%eval 0.18]` devenait `[%eval 18]`,
# soit 18 pions au lieu de 0,18. Le corpus se remplissait de fausses gaffes
# (48 % des coups evalues) sans le moindre message d'erreur. On ne nettoie donc
# plus jamais le texte avant d'avoir lu les commentaires.
JETON_RE = re.compile(r"(\{[^}]*\})|(\S+)")


def cp_depuis(txt):
    """'#-3' -> mat en 3 pour les Noirs ; '0.18' -> 18 centipions."""
    if txt.startswith("#"):
        n = int(txt[1:])
        return 10000 if n > 0 else -10000
    return int(round(float(txt) * 100))


def evals_alignees(movetext, n_coups):
    """Évaluation après chaque coup, dans l'ordre, ou None si non annoté."""
    out = []
    for com, tok in JETON_RE.findall(movetext):
        if com:
            m = EVAL_RE.search(com)
            if m and out:
                out[-1] = cp_depuis(m.group(1))
            continue
        if tok in pd.RESULT_TOKENS or pd.NAG_RE.match(tok):
            continue
        if pd.MOVENUM_RE.fullmatch(tok):
            continue
        tok = pd.ANNOTATION_RE.sub("", tok)
        if not tok:
            continue
        out.append(None)
    return out[:n_coups] + [None] * max(0, n_coups - len(out))


def traiter(bloc):
    uci, raison = pd.process_game_block(bloc, chess.Board)
    if uci is None:
        return None
    coups = uci.split()
    lignes = bloc.split("\n")
    entetes, movetext = {}, []
    for l in lignes:
        m = pd.HEADER_RE.match(l.strip())
        if m:
            entetes[m.group(1)] = m.group(2)
        elif l.strip():
            movetext.append(l)
    movetext = " ".join(movetext)
    if "[%eval" not in movetext:
        return None
    cp = evals_alignees(movetext, len(coups))
    if sum(x is not None for x in cp) < 10:
        return None
    return json.dumps({
        "uci": coups, "cp": cp,
        "res": entetes.get("Result", "*"),
        "elo": [int(entetes.get("WhiteElo", 0) or 0),
                int(entetes.get("BlackElo", 0) or 0)],
    }, separators=(",", ":"))


def blocs(chemin):
    dctx = zstandard.ZstdDecompressor()
    with open(chemin, "rb") as fh, dctx.stream_reader(fh) as flux:
        txt = io.TextIOWrapper(flux, encoding="utf-8", errors="replace")
        cur = []
        for ligne in txt:
            if ligne.startswith("[Event ") and cur:
                yield "".join(cur)
                cur = []
            cur.append(ligne)
        if cur:
            yield "".join(cur)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mois", default="2026-04")
    p.add_argument("--datadir", default="data")
    p.add_argument("--outdir", default="data/eval_pgn")
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--limite", type=int, default=0)
    args = p.parse_args()

    motif = f"{args.datadir}/lichess_db_standard_rated_{args.mois}.pgn.zst"
    src = glob.glob(motif)
    if not src:
        raise SystemExit(f"introuvable : {motif}")
    os.makedirs(args.outdir, exist_ok=True)
    out = f"{args.outdir}/eval_{args.mois}.jsonl"
    print(f"[eval] {src[0]} -> {out} ({args.workers} workers)", flush=True)

    t0 = time.time()
    lues = gardees = n_eval = 0
    with open(out, "w") as g, mp.Pool(args.workers) as pool:
        flux = blocs(src[0])
        if args.limite:
            flux = (b for _, b in zip(range(args.limite), flux))
        for res in pool.imap_unordered(traiter, flux, chunksize=64):
            lues += 1
            if res:
                g.write(res + "\n")
                gardees += 1
                n_eval += sum(1 for c in json.loads(res)["cp"] if c is not None)
            if lues % 200000 == 0:
                dt = time.time() - t0
                g.flush()
                print(f"  {lues:,} lues | {gardees:,} gardees | "
                      f"{n_eval:,} evals | {dt/60:.1f} min "
                      f"({lues/dt:.0f} parties/s)", flush=True)

    dt = time.time() - t0
    print(f"\n{lues:,} parties lues | {gardees:,} gardees ({gardees/max(lues,1):.1%})")
    print(f"{n_eval:,} evaluations recuperees -> {out}")
    print(f"duree : {dt/60:.1f} min")


if __name__ == "__main__":
    main()
