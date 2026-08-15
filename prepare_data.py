"""Phase 1 : Du dump Lichess brut à un corpus de parties en notation UCI.

Le dump mensuel de Lichess est un fichier PGN compressé en zstd d'environ 29 Go
(soit ~200 Go une fois décompressé, plus de 100 millions de parties). On veut
en extraire un sous-ensemble propre, écrit sous une forme que le modèle pourra
avaler directement : une partie par ligne, les coups en UCI séparés par des
espaces.

Deux opérations, de coûts très différents :

1. **Filtrer sur les en-têtes** : bon marché. Ce sont des comparaisons de
   chaînes sur les métadonnées de la partie (Elo des joueurs, cadence, cause de
   fin de partie).
2. **Convertir SAN vers UCI** : cher. Lichess stocke les coups en notation
   algébrique abrégée (`Nf3`), qui est *contextuelle* : « Nf3 » ne dit pas d'où
   vient le cavalier, il faut connaître la position pour le savoir. La seule
   façon de traduire en UCI (`g1f3`), qui est explicite, est de rejouer la
   partie coup par coup sur un échiquier. C'est ce qui coûte, et c'est
   incompressible.

D'où l'architecture : le processus principal ne fait que décompresser et
découper le flux en blocs de parties, puis distribue ces blocs à des processus
ouvriers qui font le filtrage *et* le rejeu. La décompression zstd est rapide
et reste séquentielle ; le rejeu, lui, est parallélisé sur les cœurs
disponibles.

Usage :
    python prepare_data.py benchmark --games 50000
    python prepare_data.py parse --workers 10 --out data/games_uci.txt
    python prepare_data.py encode --games data/games_uci.txt
"""

import argparse
import io
import json
import multiprocessing as mp
import os
import re
import sys
import time
from collections import Counter

import numpy as np
import zstandard

# --------------------------------------------------------------------------
# Critères de filtrage : ce sont les décisions figées du projet
# --------------------------------------------------------------------------

ELO_MIN = 1800
ELO_MAX = 2600
MIN_PLIES = 20          # demi-coups : en dessous, la partie n'a rien à apprendre
MAX_PLIES = 300         # au-dessus, c'est une anomalie (et ça dépasse le contexte)
EXCLUDED_SPEEDS = ("Bullet", "UltraBullet")
REQUIRED_TERMINATION = "Normal"

# --------------------------------------------------------------------------
# Extraction du texte des coups
# --------------------------------------------------------------------------

# Les commentaires Lichess : { [%eval 0.17] [%clk 0:03:00] }
COMMENT_RE = re.compile(r"\{[^}]*\}")
# Les numéros de coup : "1." pour les blancs, "1..." pour les noirs
MOVENUM_RE = re.compile(r"\d+\.+")
# Les résultats en fin de ligne
RESULT_TOKENS = {"1-0", "0-1", "1/2-1/2", "*"}
# Les annotations d'appréciation collées au coup : Nf3!? -> Nf3
ANNOTATION_RE = re.compile(r"[?!]+$")
# Les NAG numériques : $1, $18...
NAG_RE = re.compile(r"^\$\d+$")

HEADER_RE = re.compile(r'^\[(\w+)\s+"(.*)"\]$')


def extract_san_moves(movetext: str) -> list[str]:
    """Transforme le texte PGN des coups en une liste de coups SAN nus."""
    movetext = COMMENT_RE.sub(" ", movetext)
    movetext = MOVENUM_RE.sub(" ", movetext)
    moves = []
    for token in movetext.split():
        if token in RESULT_TOKENS or NAG_RE.match(token):
            continue
        token = ANNOTATION_RE.sub("", token)
        if token:
            moves.append(token)
    return moves


# --------------------------------------------------------------------------
# Traitement d'une partie (exécuté dans les processus ouvriers)
# --------------------------------------------------------------------------

# Codes de rejet, pour pouvoir expliquer le taux de rétention plutôt que de le
# constater. Chaque partie écartée l'est pour exactement une raison, la
# première rencontrée dans l'ordre ci-dessous.
REJECT_REASONS = [
    "cadence_exclue",       # bullet ou ultrabullet
    "elo_hors_bornes",      # au moins un joueur hors [1800, 2600]
    "elo_absent",           # en-tête WhiteElo/BlackElo manquant ou non numérique
    "terminaison",          # Termination != Normal
    "position_initiale",    # partie démarrée depuis une position custom (FEN)
    "trop_courte",          # < 20 demi-coups
    "trop_longue",          # > 300 demi-coups
    "san_invalide",         # coup impossible à rejouer (PGN corrompu)
]


def process_game_block(block: str, board_cls):
    """Filtre puis convertit une partie. Renvoie (uci_string | None, raison).

    `board_cls` est passé en argument plutôt qu'importé au niveau module pour
    que l'import de python-chess n'ait lieu qu'une fois par ouvrier.
    """
    headers = {}
    movetext_lines = []
    in_movetext = False

    for line in block.split("\n"):
        line = line.strip()
        if not line:
            if headers:
                in_movetext = True
            continue
        if not in_movetext and line.startswith("["):
            m = HEADER_RE.match(line)
            if m:
                headers[m.group(1)] = m.group(2)
        else:
            in_movetext = True
            movetext_lines.append(line)

    # --- Filtres bon marché, du plus discriminant au moins discriminant ---

    event = headers.get("Event", "")
    if any(speed in event for speed in EXCLUDED_SPEEDS):
        return None, "cadence_exclue"

    try:
        white_elo = int(headers["WhiteElo"])
        black_elo = int(headers["BlackElo"])
    except (KeyError, ValueError):
        return None, "elo_absent"

    if not (ELO_MIN <= white_elo <= ELO_MAX and ELO_MIN <= black_elo <= ELO_MAX):
        return None, "elo_hors_bornes"

    if headers.get("Termination") != REQUIRED_TERMINATION:
        return None, "terminaison"

    # Une partie démarrée depuis une position arbitraire fausserait
    # l'apprentissage : le modèle ne voit pas la position, seulement les coups,
    # donc il croirait que ces coups suivent le début de partie standard.
    if "FEN" in headers:
        return None, "position_initiale"

    san_moves = extract_san_moves(" ".join(movetext_lines))

    if len(san_moves) < MIN_PLIES:
        return None, "trop_courte"
    if len(san_moves) > MAX_PLIES:
        return None, "trop_longue"

    # --- Filtre cher : le rejeu ---

    board = board_cls()
    uci_moves = []
    try:
        for san in san_moves:
            move = board.parse_san(san)
            uci_moves.append(move.uci())
            board.push(move)
    except Exception:
        return None, "san_invalide"

    return " ".join(uci_moves), None


# --------------------------------------------------------------------------
# Ouvriers
# --------------------------------------------------------------------------

_BOARD_CLS = None


def _worker_init():
    """Importe python-chess une seule fois par processus ouvrier."""
    global _BOARD_CLS
    import chess
    _BOARD_CLS = chess.Board


def _worker_process_batch(blocks: list[str]):
    """Traite un lot de parties.

    Renvoie (lignes_uci, compteur_de_rejets, nombre_de_parties_lues). Le
    troisième élément peut sembler redondant avec la taille du lot, mais le
    dernier lot d'un flux est presque toujours incomplet : sans ce compteur,
    le taux de rétention publié serait faux de quelques dixièmes de pourcent.
    """
    kept = []
    rejected = Counter()
    for block in blocks:
        uci, reason = process_game_block(block, _BOARD_CLS)
        if uci is None:
            rejected[reason] += 1
        else:
            kept.append(uci)
    return kept, rejected, len(blocks)


# --------------------------------------------------------------------------
# Lecture du flux compressé
# --------------------------------------------------------------------------

def iter_game_blocks(path: str, read_size: int = 1 << 24):
    """Parcourt le .pgn.zst et produit un bloc de texte par partie.

    On ne décompresse jamais le fichier sur disque : `stream_reader` nous donne
    un flux, qu'on lit par tranches de 16 Mo. C'est ce qui permet de traiter
    200 Go de PGN avec 29 Go sur le disque et quelques centaines de Mo de RAM.

    Le découpage se fait sur la balise `[Event `, qui ouvre chaque partie.
    """
    dctx = zstandard.ZstdDecompressor()
    with open(path, "rb") as fh:
        with dctx.stream_reader(fh, read_size=read_size) as reader:
            text_stream = io.TextIOWrapper(reader, encoding="utf-8",
                                           errors="replace")
            current = []
            for line in text_stream:
                if line.startswith("[Event ") and current:
                    yield "".join(current)
                    current = [line]
                else:
                    current.append(line)
            if current:
                yield "".join(current)


def batched(iterable, size):
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


# --------------------------------------------------------------------------
# Commande : benchmark
# --------------------------------------------------------------------------

def cmd_benchmark(args):
    """Mesure le débit de parsing, en séquentiel et en parallèle.

    L'objectif est de répondre à une question concrète : combien de temps pour
    atteindre ~800 M de tokens de coups ? Si la réponse dépasse 8 heures en
    séquentiel, on parallélise.
    """
    import chess

    print(f"Lecture de {args.games} parties depuis {args.input}\n")

    # --- 1. Coût de la lecture seule (décompression + découpage) ---
    t0 = time.perf_counter()
    blocks = []
    for i, block in enumerate(iter_game_blocks(args.input)):
        blocks.append(block)
        if i + 1 >= args.games:
            break
    t_read = time.perf_counter() - t0
    print(f"[1] Décompression + découpage : {t_read:.1f} s "
          f"({len(blocks)/t_read:,.0f} parties/s)")

    # --- 2. Coût du traitement complet, un seul cœur ---
    t0 = time.perf_counter()
    kept_seq = []
    rejected = Counter()
    for block in blocks:
        uci, reason = process_game_block(block, chess.Board)
        if uci is None:
            rejected[reason] += 1
        else:
            kept_seq.append(uci)
    t_seq = time.perf_counter() - t0
    rate_seq = len(blocks) / t_seq
    print(f"[2] Filtrage + rejeu SAN->UCI, 1 cœur : {t_seq:.1f} s "
          f"({rate_seq:,.0f} parties/s)")

    # --- 3. Même chose, en parallèle ---
    workers = args.workers
    t0 = time.perf_counter()
    with mp.Pool(workers, initializer=_worker_init) as pool:
        results = pool.map(_worker_process_batch,
                           list(batched(blocks, 500)))
    t_par = time.perf_counter() - t0
    kept_par = sum(len(k) for k, _, _ in results)
    rate_par = len(blocks) / t_par
    print(f"[3] Idem sur {workers} ouvriers : {t_par:.1f} s "
          f"({rate_par:,.0f} parties/s, accélération x{rate_par/rate_seq:.1f})")

    assert kept_par == len(kept_seq), "séquentiel et parallèle divergent !"

    # --- Statistiques et extrapolation ---
    n_kept = len(kept_seq)
    retention = n_kept / len(blocks)
    lengths = [len(g.split()) for g in kept_seq]
    mean_len = sum(lengths) / len(lengths) if lengths else 0

    print(f"\n--- Sur cet échantillon de {len(blocks):,} parties ---")
    print(f"  conservées        : {n_kept:,} ({retention:.1%})")
    print(f"  longueur moyenne  : {mean_len:.1f} demi-coups")
    print(f"  tokens par partie lue : {mean_len * retention:.1f}")
    print(f"  motifs de rejet :")
    for reason, count in rejected.most_common():
        print(f"    {reason:<20} {count:>8,} ({count/len(blocks):5.1%})")

    target_tokens = args.target_tokens
    tokens_per_read_game = mean_len * retention
    if tokens_per_read_game > 0:
        games_needed = target_tokens / tokens_per_read_game
        games_kept_needed = games_needed * retention
        print(f"\n--- Extrapolation pour {target_tokens/1e6:.0f} M de tokens ---")
        print(f"  parties à lire      : {games_needed/1e6:.1f} M")
        print(f"  parties conservées  : {games_kept_needed/1e6:.1f} M")
        print(f"  durée 1 cœur        : {games_needed/rate_seq/3600:.1f} h")
        print(f"  durée {workers} ouvriers : "
              f"{games_needed/rate_par/3600:.1f} h")

    report = {
        "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "methode": ("lecture des N premières parties du dump, chronométrage "
                    "séparé de la décompression, du traitement séquentiel et "
                    "du traitement parallèle sur le même échantillon"),
        "echantillon_parties_lues": len(blocks),
        "cpu_workers": workers,
        "decompression_parties_s": round(len(blocks) / t_read, 1),
        "sequentiel_parties_s": round(rate_seq, 1),
        "parallele_parties_s": round(rate_par, 1),
        "acceleration": round(rate_par / rate_seq, 2),
        "taux_retention": round(retention, 4),
        "longueur_moyenne_demi_coups": round(mean_len, 2),
        "rejets": dict(rejected),
        "extrapolation_target_tokens": target_tokens,
        "extrapolation_parties_a_lire_millions": round(games_needed / 1e6, 2),
        "extrapolation_heures_sequentiel": round(games_needed / rate_seq / 3600, 2),
        "extrapolation_heures_parallele": round(games_needed / rate_par / 3600, 2),
    }
    with open(args.report, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nRapport écrit dans {args.report}")


# --------------------------------------------------------------------------
# Commande : parse
# --------------------------------------------------------------------------

def cmd_parse(args):
    """Parcourt tout le dump et écrit le corpus UCI."""
    t_start = time.perf_counter()
    n_read = 0
    n_kept = 0
    rejected = Counter()
    length_hist = Counter()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    print(f"Entrée  : {args.input}")
    print(f"Sortie  : {args.out}")
    print(f"Ouvriers: {args.workers}")
    print(f"Objectif: {args.target_tokens/1e6:.0f} M de tokens "
          f"({'illimité' if args.max_games == 0 else f'{args.max_games:,} parties max'})\n",
          flush=True)

    n_tokens = 0
    stop = False
    dump_exhausted = True

    # On traite le flux par fenêtres plutôt qu'en le passant d'un bloc au pool.
    # `Pool.imap` consomme son itérable d'entrée aussi vite qu'il le peut et
    # met en file d'attente *toutes* les tâches : sur 92 millions de parties,
    # ça représente une centaine de gigaoctets de texte en RAM et un processus
    # tué par l'OOM killer au bout de quelques minutes. En découpant en
    # fenêtres de taille fixe traitées l'une après l'autre, l'empreinte mémoire
    # reste bornée par la fenêtre, ici quelques dizaines de mégaoctets.
    window_batches = args.window // args.batch_size

    with open(args.out, "w") as out_fh, \
         mp.Pool(args.workers, initializer=_worker_init) as pool:

        batch_stream = batched(iter_game_blocks(args.input), args.batch_size)

        while not stop:
            window = []
            for batch in batch_stream:
                window.append(batch)
                if len(window) >= window_batches:
                    break
            if not window:
                break                      # flux épuisé
            if len(window) < window_batches:
                dump_exhausted = True      # dernière fenêtre, incomplète

            for kept, rej, n_blocks in pool.map(_worker_process_batch, window):
                n_read += n_blocks
                n_kept += len(kept)
                rejected.update(rej)
                for game in kept:
                    n_moves = game.count(" ") + 1
                    length_hist[n_moves] += 1
                    n_tokens += n_moves
                    out_fh.write(game)
                    out_fh.write("\n")

            elapsed = time.perf_counter() - t_start
            print(f"  {n_read/1e6:6.2f} M lues | {n_kept/1e6:5.2f} M gardées "
                  f"({n_kept/max(n_read,1):5.1%}) | {n_tokens/1e6:7.1f} M tokens "
                  f"| {n_read/elapsed:,.0f} parties/s "
                  f"| {elapsed/60:5.1f} min", flush=True)

            if args.target_tokens and n_tokens >= args.target_tokens:
                print(f"\nObjectif de {args.target_tokens/1e6:.0f} M tokens atteint.")
                dump_exhausted = False
                stop = True
            if args.max_games and n_read >= args.max_games:
                dump_exhausted = False
                stop = True

    elapsed = time.perf_counter() - t_start
    mean_len = n_tokens / n_kept if n_kept else 0

    stats = {
        "dump_epuise_avant_objectif": dump_exhausted,
        "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": os.path.basename(args.input),
        "parties_lues": n_read,
        "parties_conservees": n_kept,
        "taux_retention": round(n_kept / max(n_read, 1), 4),
        "tokens_coups_total": n_tokens,
        "longueur_moyenne_demi_coups": round(mean_len, 2),
        "duree_secondes": round(elapsed, 1),
        "debit_parties_s": round(n_read / elapsed, 1),
        "workers": args.workers,
        "rejets": dict(rejected),
        "histogramme_longueurs": dict(sorted(length_hist.items())),
        "criteres": {
            "elo_min": ELO_MIN, "elo_max": ELO_MAX,
            "min_demi_coups": MIN_PLIES, "max_demi_coups": MAX_PLIES,
            "cadences_exclues": list(EXCLUDED_SPEEDS),
            "terminaison_requise": REQUIRED_TERMINATION,
        },
    }
    with open(args.stats, "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"\n=== Terminé en {elapsed/60:.1f} min ===")
    print(f"  parties lues       : {n_read:,}")
    print(f"  parties conservées : {n_kept:,} ({n_kept/max(n_read,1):.1%})")
    print(f"  tokens de coups    : {n_tokens:,}")
    print(f"  longueur moyenne   : {mean_len:.1f} demi-coups")
    print(f"  statistiques -> {args.stats}")


# --------------------------------------------------------------------------
# Commande : encode
# --------------------------------------------------------------------------

def cmd_encode(args):
    """Encode le corpus texte en deux .bin uint16 (train et val).

    Pourquoi uint16 : le vocabulaire fait 1971 entrées, qui tiennent largement
    sur 16 bits. Stocker en uint16 plutôt qu'en int32 divise par deux la taille
    du fichier et donc le volume à faire transiter depuis le disque pendant
    l'entraînement. À 800 M de tokens, c'est 1.6 Go au lieu de 3.2 Go, assez
    petit pour que le cache disque du système garde tout en RAM.

    Chaque partie est encodée <bos> coup1 coup2 ... coupN <eos>, et toutes les
    parties sont concaténées bout à bout dans un seul tableau plat. C'est le
    format le plus simple à échantillonner : pour un batch, on tire des
    positions au hasard dans le tableau et on lit 256 tokens consécutifs.
    """
    with open(args.vocab) as f:
        vocab = json.load(f)
    stoi = vocab["stoi"]
    bos, eos = vocab["bos_id"], vocab["eos_id"]

    print(f"Vocabulaire : {vocab['vocab_size']} tokens")
    print(f"Corpus      : {args.games}")

    # Premier passage : compter les tokens pour dimensionner le memmap.
    print("Passage 1/2, comptage...", flush=True)
    t0 = time.perf_counter()
    n_games = 0
    n_tokens = 0
    with open(args.games) as f:
        for line in f:
            n_moves = line.count(" ") + 1
            n_tokens += n_moves + 2   # <bos> et <eos>
            n_games += 1
    print(f"  {n_games:,} parties, {n_tokens:,} tokens "
          f"({time.perf_counter()-t0:.1f} s)")

    n_val_games = max(1, int(n_games * args.val_fraction))
    n_train_games = n_games - n_val_games
    print(f"  découpage : {n_train_games:,} parties d'entraînement, "
          f"{n_val_games:,} de validation ({args.val_fraction:.1%})")

    # Le découpage se fait par *partie* et non par token : une partie ne doit
    # jamais être coupée entre train et val, sinon le modèle aurait vu le début
    # d'une partie qu'on lui demande ensuite de prédire. C'est une fuite de
    # données classique, et sournoise parce qu'elle améliore la loss de
    # validation sans améliorer le modèle.
    rng = np.random.default_rng(args.seed)
    is_val = np.zeros(n_games, dtype=bool)
    is_val[rng.choice(n_games, size=n_val_games, replace=False)] = True

    print("Passage 2/2, encodage...", flush=True)
    t0 = time.perf_counter()

    train_path = os.path.join(args.outdir, "train.bin")
    val_path = os.path.join(args.outdir, "val.bin")
    # On alloue large puis on tronque : on connaît n_tokens exactement, mais on
    # ne sait pas encore comment il se répartit entre train et val.
    train_buf = np.memmap(train_path, dtype=np.uint16, mode="w+", shape=(n_tokens,))
    val_buf = np.memmap(val_path, dtype=np.uint16, mode="w+", shape=(n_tokens,))

    # On conserve aussi les parties de validation en clair. L'évaluation de la
    # phase 4 en a besoin : pour vérifier la légalité d'un coup il faut rejouer
    # la position sur un échiquier, ce qui demande les coups en UCI et non
    # leurs identifiants numériques.
    val_txt_path = os.path.join(args.outdir, "val_games.txt")

    ti = vi = 0
    unknown = Counter()
    with open(args.games) as f, open(val_txt_path, "w") as val_txt:
        for idx, line in enumerate(f):
            moves = line.split()
            try:
                ids = [stoi[m] for m in moves]
            except KeyError as e:
                unknown[str(e)] += 1
                continue
            seq = np.fromiter([bos] + ids + [eos], dtype=np.uint16,
                              count=len(ids) + 2)
            if is_val[idx]:
                val_buf[vi:vi + len(seq)] = seq
                vi += len(seq)
                val_txt.write(line if line.endswith("\n") else line + "\n")
            else:
                train_buf[ti:ti + len(seq)] = seq
                ti += len(seq)

            if (idx + 1) % 1_000_000 == 0:
                print(f"  {idx+1:,} parties encodées "
                      f"({time.perf_counter()-t0:.0f} s)", flush=True)

    train_buf.flush()
    val_buf.flush()
    del train_buf, val_buf

    # Troncature aux tailles réelles
    with open(train_path, "r+b") as f:
        f.truncate(ti * 2)
    with open(val_path, "r+b") as f:
        f.truncate(vi * 2)

    assert not unknown, f"coups absents du vocabulaire : {unknown}"

    stats = {
        "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "vocab_size": vocab["vocab_size"],
        "parties_total": n_games,
        "parties_train": n_train_games,
        "parties_val": n_val_games,
        "tokens_train": int(ti),
        "tokens_val": int(vi),
        "tokens_total": int(ti + vi),
        "val_fraction_parties": args.val_fraction,
        "seed_split": args.seed,
        "octets_train": int(ti * 2),
        "octets_val": int(vi * 2),
        "dtype": "uint16",
        "format": "<bos> coups... <eos>, parties concaténées bout à bout",
        "val_games_txt": os.path.abspath(val_txt_path),
    }
    with open(os.path.join(args.outdir, "encode_stats.json"), "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"\n=== Encodage terminé en {(time.perf_counter()-t0)/60:.1f} min ===")
    print(f"  train.bin : {ti:,} tokens ({ti*2/1e9:.2f} Go)")
    print(f"  val.bin   : {vi:,} tokens ({vi*2/1e9:.2f} Go)")


# --------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    default_input = "data/lichess_db_standard_rated_2026-07.pgn.zst"

    b = sub.add_parser("benchmark", help="mesure le débit de parsing")
    b.add_argument("--input", default=default_input)
    b.add_argument("--games", type=int, default=50_000)
    b.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 2))
    b.add_argument("--target-tokens", type=float, default=800e6)
    b.add_argument("--report", default="logs/phase1_benchmark.json")
    b.set_defaults(func=cmd_benchmark)

    q = sub.add_parser("parse", help="traite tout le dump")
    q.add_argument("--input", default=default_input)
    q.add_argument("--out", default="data/games_uci.txt")
    q.add_argument("--stats", default="logs/phase1_parse_stats.json")
    q.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 2))
    q.add_argument("--batch-size", type=int, default=500)
    q.add_argument("--window", type=int, default=200_000,
                   help="parties chargées en RAM à la fois (borne mémoire)")
    q.add_argument("--target-tokens", type=float, default=800e6)
    q.add_argument("--max-games", type=int, default=0)
    q.set_defaults(func=cmd_parse)

    e = sub.add_parser("encode", help="encode en .bin uint16")
    e.add_argument("--games", default="data/games_uci.txt")
    e.add_argument("--vocab", default="data/vocab.json")
    e.add_argument("--outdir", default="data")
    e.add_argument("--val-fraction", type=float, default=0.01)
    e.add_argument("--seed", type=int, default=1337)
    e.set_defaults(func=cmd_encode)

    args = p.parse_args()
    args.target_tokens = int(getattr(args, "target_tokens", 0) or 0)
    args.func(args)


if __name__ == "__main__":
    main()
