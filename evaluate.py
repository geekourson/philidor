"""Phase 4 : Mesurer ce que le modèle a réellement appris.

Le chiffre central du projet est le **taux de coups légaux en génération
libre** : sur une position donnée, sans aucune contrainte imposée, quelle
proportion des coups que le modèle propose sont effectivement jouables ?

Ce chiffre mérite qu'on insiste sur ce qu'il mesure. Le modèle n'a jamais vu
d'échiquier, ne connaît pas les règles, et n'a aucun moyen de vérifier quoi que
ce soit. S'il propose un coup légal, c'est uniquement parce qu'il a inféré les
règles du jeu à partir de la seule régularité statistique de millions de
parties. Un taux élevé n'est donc pas une performance d'échecs : c'est la
preuve qu'un système entraîné à prédire des symboles a reconstruit une
mécanique sous-jacente qu'on ne lui a jamais décrite.

Toutes les mesures sont accompagnées d'un intervalle de confiance, parce qu'un
taux mesuré sur un échantillon fini est une estimation, pas une vérité.
"""

import argparse
import json
import math
import os
import random
import time
from collections import Counter

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import chess
import torch
import torch.nn.functional as F

from model import ChessGPT, ModelConfig


# ---------------------------------------------------------------------------
# Statistiques
# ---------------------------------------------------------------------------

def wilson_interval(successes: int, total: int, z: float = 1.96):
    """Intervalle de confiance de Wilson à 95 %.

    Pourquoi Wilson plutôt que l'intervalle « normal » enseigné partout
    (p ± 1.96 √(p(1-p)/n)) : ce dernier se comporte très mal quand le taux
    approche 0 ou 100 %. Il peut produire une borne supérieure au-dessus de
    100 %, ou un intervalle de largeur nulle si aucun échec n'est observé, ce
    qui affirmerait une certitude absolue à partir d'un échantillon fini.
    Comme on espère précisément mesurer des taux proches de 100 %, c'est le
    cas dégénéré qui nous concerne directement.
    """
    if total == 0:
        return (0.0, 0.0, 0.0)
    p = successes / total
    denom = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    margin = (z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
              / denom)
    return p, max(0.0, centre - margin), min(1.0, centre + margin)


# ---------------------------------------------------------------------------
# Chargement
# ---------------------------------------------------------------------------

def load_model(ckpt_path: str, device: str):
    """Charge un checkpoint de reprise ou un instantané léger indifféremment.

    Les deux contiennent les poids, la configuration du modèle et les
    compteurs ; seul le checkpoint de reprise porte en plus l'état de
    l'optimiseur, dont l'évaluation n'a que faire.
    """
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ModelConfig(**ck["model_config"])
    model = ChessGPT(cfg).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, ck


def load_vocab(path: str):
    with open(path) as f:
        v = json.load(f)
    return v["stoi"], v["itos"], v["bos_id"], v["eos_id"]


@torch.no_grad()
def last_logits_batched(model, prompts, device, batch_size, block_size):
    """Logits du dernier token, pour une liste de préfixes de longueurs variées.

    Subtilité qui coûte cher si on la rate. La façon naturelle de traiter des
    séquences de longueurs différentes en un seul batch est de les compléter
    par un token de remplissage. Mais notre modèle n'a pas de masque de
    remplissage : son attention est purement causale, donc chaque position
    regarde *tout* ce qui la précède, y compris le remplissage. Compléter à
    gauche avec des <bos> revient donc à faire croire au modèle que la partie a
    commencé plusieurs fois : ce qui pollue la prédiction.

    On regroupe donc les préfixes par longueur exacte : aucun remplissage,
    aucun token parasite, et on garde le bénéfice du traitement par lots.
    C'est d'autant plus important que le taux de coups légaux est le chiffre
    central du projet : une mesure biaisée par un artefact de remplissage
    n'aurait aucune valeur.
    """
    by_length = {}
    for i, p in enumerate(prompts):
        p = p[-block_size:]
        by_length.setdefault(len(p), []).append((i, p))

    out = [None] * len(prompts)
    for length, items in by_length.items():
        for start in range(0, len(items), batch_size):
            chunk = items[start:start + batch_size]
            batch = torch.tensor([p for _, p in chunk], dtype=torch.long,
                                 device=device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, _ = model(batch)
            logits = logits[:, -1, :].float()
            for row, (idx, _) in enumerate(chunk):
                out[idx] = logits[row]
    return torch.stack(out)


# ---------------------------------------------------------------------------
# 1. Taux de coups légaux en génération libre
# ---------------------------------------------------------------------------

@torch.no_grad()
def legal_move_rate(model, stoi, itos, bos, games, device, n_positions,
                    temperature, batch_size, rng):
    """Sur des positions issues de vraies parties, un coup échantillonné
    librement est-il légal ?

    Protocole : on prend une partie du jeu de validation, on la rejoue jusqu'à
    un demi-coup tiré au hasard, on donne au modèle la séquence des coups
    joués, et on échantillonne **un** coup dans sa distribution de sortie sans
    aucun masquage. On vérifie ensuite la légalité avec python-chess.

    Aucune contrainte n'est appliquée : le modèle a le droit de proposer
    n'importe lequel des 1971 tokens du vocabulaire, y compris <pad>.
    """
    prompts, boards = [], []
    for _ in range(n_positions):
        game = rng.choice(games)
        moves = game.split()
        # On coupe entre le 1er et l'avant-dernier demi-coup
        cut = rng.randint(1, len(moves) - 1)
        board = chess.Board()
        for m in moves[:cut]:
            board.push(chess.Move.from_uci(m))
        prompts.append([bos] + [stoi[m] for m in moves[:cut]])
        boards.append(board)

    legal = 0
    illegal_reasons = Counter()

    all_logits = last_logits_batched(model, prompts, device, batch_size,
                                     model.cfg.block_size)

    if temperature == 0.0:
        ids = all_logits.argmax(dim=-1)
    else:
        probs = F.softmax(all_logits / temperature, dim=-1)
        ids = torch.multinomial(probs, num_samples=1).squeeze(-1)

    for i, tok_id in enumerate(ids.tolist()):
        token = itos[tok_id]
        if token in ("<pad>", "<bos>", "<eos>"):
            illegal_reasons["token_special"] += 1
            continue
        try:
            move = chess.Move.from_uci(token)
        except ValueError:
            illegal_reasons["uci_invalide"] += 1
            continue
        if move in boards[i].legal_moves:
            legal += 1
        else:
            # On distingue plusieurs échecs très différents : viser une case
            # vide, déplacer une pièce adverse, ou proposer un coup
            # géométriquement correct mais illégal dans cette position (roi
            # en échec, pièce clouée, case d'arrivée occupée par un allié).
            if boards[i].piece_at(move.from_square) is None:
                illegal_reasons["case_depart_vide"] += 1
            elif (boards[i].piece_at(move.from_square).color
                  != boards[i].turn):
                illegal_reasons["piece_adverse"] += 1
            else:
                illegal_reasons["deplacement_illegal"] += 1

    total = len(prompts)
    p, lo, hi = wilson_interval(legal, total)
    return {
        "positions_testees": total,
        "coups_legaux": legal,
        "taux": round(p, 5),
        "ic95_bas": round(lo, 5),
        "ic95_haut": round(hi, 5),
        "temperature": temperature,
        "motifs_echec": dict(illegal_reasons),
        "methode": ("un coup échantillonné sans masquage sur chaque position, "
                    "positions tirées à un demi-coup aléatoire de parties du "
                    "jeu de validation, légalité vérifiée par python-chess ; "
                    "intervalle de Wilson à 95 %"),
    }


# ---------------------------------------------------------------------------
# 2. Parties complètes sans aucun coup illégal
# ---------------------------------------------------------------------------

@torch.no_grad()
def full_game_rate(model, stoi, itos, bos, eos, device, n_games,
                   temperature, max_plies, rng, batch_size=64):
    """Le modèle joue des parties entières contre lui-même, depuis la position
    de départ, sans jamais être corrigé. Combien vont au bout sans faute ?

    C'est une mesure bien plus sévère que le taux par coup. Une partie de 80
    demi-coups n'est comptée comme réussie que si les 80 coups sont légaux.
    Même avec un taux par coup de 99 %, seules 45 % des parties passeraient
    (0.99^80). Ce chiffre teste donc autant la constance que la compétence.

    Les parties sont jouées **en parallèle**, par paquets. La version naïve -
    une partie après l'autre, enchaîne jusqu'à 250 passes avant successives
    avec une seule séquence à la fois, ce qui laisse le GPU à 20 % d'occupation :
    on paie le coût de lancement d'un noyau CUDA pour calculer un unique coup.
    En jouant 64 parties de front, la carte travaille sur 64 séquences à chaque
    passe et le temps d'évaluation s'effondre.

    Une subtilité rend ce parallélisme facile ici : toutes les parties du
    paquet démarrent au même moment et gagnent exactement un token par tour.
    Elles ont donc toujours la même longueur, et aucun remplissage n'est
    nécessaire. Les parties terminées avant les autres restent dans le tenseur
   , on ignore simplement ce qu'elles produisent.
    """
    successes = 0
    plies_before_error = []
    lengths = []
    terminations = Counter()
    block = model.cfg.block_size

    remaining = n_games
    while remaining > 0:
        b = min(batch_size, remaining)
        remaining -= b

        boards = [chess.Board() for _ in range(b)]
        seqs = torch.full((b, 1), bos, dtype=torch.long, device=device)
        active = [True] * b
        ok = [True] * b
        done_len = [0] * b

        for ply in range(max_plies):
            if not any(active):
                break
            ctx = seqs[:, -block:]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, _ = model(ctx)
            logits = logits[:, -1, :].float()

            if temperature == 0.0:
                ids = logits.argmax(dim=-1)
            else:
                probs = F.softmax(logits / temperature, dim=-1)
                ids = torch.multinomial(probs, num_samples=1).squeeze(-1)
            ids_list = ids.tolist()

            for i in range(b):
                if not active[i]:
                    continue
                token = itos[ids_list[i]]

                if token == "<eos>":
                    terminations["eos_predit"] += 1
                    active[i] = False
                    done_len[i] = ply
                    continue
                if token in ("<pad>", "<bos>"):
                    ok[i] = False
                    plies_before_error.append(ply)
                    terminations["token_special"] += 1
                    active[i] = False
                    done_len[i] = ply
                    continue
                try:
                    move = chess.Move.from_uci(token)
                except ValueError:
                    ok[i] = False
                    plies_before_error.append(ply)
                    terminations["uci_invalide"] += 1
                    active[i] = False
                    done_len[i] = ply
                    continue
                if move not in boards[i].legal_moves:
                    ok[i] = False
                    plies_before_error.append(ply)
                    terminations["coup_illegal"] += 1
                    active[i] = False
                    done_len[i] = ply
                    continue

                boards[i].push(move)
                if boards[i].is_game_over():
                    terminations["partie_terminee_regles"] += 1
                    active[i] = False
                    done_len[i] = ply + 1

            # Les parties terminées continuent de recevoir le token prédit :
            # leur contenu n'est plus lu, seule la forme du tenseur compte.
            seqs = torch.cat([seqs, ids.unsqueeze(1)], dim=1)

        for i in range(b):
            if active[i]:
                terminations["longueur_max_atteinte"] += 1
                done_len[i] = max_plies
            if ok[i]:
                successes += 1
            lengths.append(done_len[i])

    p, lo, hi = wilson_interval(successes, n_games)
    mean_err = (sum(plies_before_error) / len(plies_before_error)
                if plies_before_error else None)
    return {
        "parties_jouees": n_games,
        "parties_sans_coup_illegal": successes,
        "taux": round(p, 5),
        "ic95_bas": round(lo, 5),
        "ic95_haut": round(hi, 5),
        "temperature": temperature,
        "longueur_moyenne_demi_coups": round(sum(lengths) / len(lengths), 2),
        "demi_coups_moyens_avant_premiere_faute": (
            round(mean_err, 2) if mean_err is not None else "aucune faute"),
        "terminaisons": dict(terminations),
        "methode": ("parties générées depuis la position initiale, le modèle "
                    "joue les deux couleurs, aucun masquage, arrêt à la "
                    "première illégalité ou à <eos>"),
    }


# ---------------------------------------------------------------------------
# 3. Accord avec le coup humain
# ---------------------------------------------------------------------------

@torch.no_grad()
def human_agreement(model, stoi, itos, bos, games, device, n_positions,
                    batch_size, rng):
    """Le modèle propose-t-il le coup que l'humain a réellement joué ?

    Attention à l'interprétation : ce n'est pas une mesure de la qualité du
    jeu. Un moteur d'échecs parfait aurait un accord *faible* avec des joueurs
    de 2000 Elo, puisqu'il jouerait mieux qu'eux. Ce chiffre mesure la fidélité
    à la distribution d'entraînement, ce qui est exactement l'objectif d'un
    modèle de langage, mais ne dit rien de la force au jeu, que seule la
    calibration Elo de la phase 5 pourra établir.
    """
    prompts, targets = [], []
    for _ in range(n_positions):
        game = rng.choice(games)
        moves = game.split()
        cut = rng.randint(1, len(moves) - 1)
        prompts.append([bos] + [stoi[m] for m in moves[:cut]])
        targets.append(stoi[moves[cut]])

    top1 = top5 = 0
    all_logits = last_logits_batched(model, prompts, device, batch_size,
                                     model.cfg.block_size)
    top = all_logits.topk(5, dim=-1).indices.tolist()
    for i, t in enumerate(targets):
        if top[i][0] == t:
            top1 += 1
        if t in top[i]:
            top5 += 1

    p1, lo1, hi1 = wilson_interval(top1, n_positions)
    p5, lo5, hi5 = wilson_interval(top5, n_positions)
    return {
        "positions_testees": n_positions,
        "top1": {"taux": round(p1, 5), "ic95_bas": round(lo1, 5),
                 "ic95_haut": round(hi1, 5)},
        "top5": {"taux": round(p5, 5), "ic95_bas": round(lo5, 5),
                 "ic95_haut": round(hi5, 5)},
        "methode": ("comparaison du coup le plus probable (et des 5 plus "
                    "probables) au coup effectivement joué par l'humain, sur "
                    "des positions du jeu de validation ; intervalle de Wilson"),
    }


# ---------------------------------------------------------------------------
# 4. Tests par règle
# ---------------------------------------------------------------------------

def mine_rule_positions(games, stoi, bos, n_per_rule, rng, max_scan):
    """Extrait du corpus des positions illustrant chaque règle particulière.

    Plutôt que d'inventer des positions, on les prend dans de vraies parties :
    on cherche les moments où un humain a roqué, pris en passant, promu, ou
    joué alors qu'il était en échec. On dispose ainsi, pour chaque règle, de
    positions où le coup en question est non seulement légal mais pertinent.
    """
    buckets = {"roque": [], "en_passant": [], "promotion": [], "echec": []}
    scanned = 0

    for game in games:
        if scanned >= max_scan:
            break
        if all(len(v) >= n_per_rule for v in buckets.values()):
            break
        scanned += 1
        moves = game.split()
        board = chess.Board()
        prefix = [bos]
        for i, uci in enumerate(moves):
            move = chess.Move.from_uci(uci)
            entry = None
            if board.is_check() and len(buckets["echec"]) < n_per_rule:
                entry = ("echec", list(prefix), uci)
            elif (board.is_castling(move)
                  and len(buckets["roque"]) < n_per_rule):
                entry = ("roque", list(prefix), uci)
            elif (board.is_en_passant(move)
                  and len(buckets["en_passant"]) < n_per_rule):
                entry = ("en_passant", list(prefix), uci)
            elif move.promotion and len(buckets["promotion"]) < n_per_rule:
                entry = ("promotion", list(prefix), uci)

            if entry is not None and len(prefix) > 1:
                buckets[entry[0]].append(
                    {"prefix": entry[1], "human_move": entry[2],
                     "fen": board.fen()})

            prefix.append(stoi[uci])
            board.push(move)

    return buckets, scanned


@torch.no_grad()
def rule_tests(model, stoi, itos, bos, games, device, n_per_rule, rng,
               max_scan):
    """Pour chaque règle, mesure trois choses sur les positions extraites :

    - le coup le plus probable du modèle est-il légal ?
    - correspond-il au coup humain ?
    - la probabilité totale que le modèle place sur les coups légaux
      (une mesure plus fine que le simple top-1 : elle dit si le modèle a
      compris la contrainte ou s'il a seulement de la chance)
    """
    buckets, scanned = mine_rule_positions(games, stoi, bos, n_per_rule, rng,
                                           max_scan)
    results = {"_parties_parcourues": scanned}
    block = model.cfg.block_size

    for rule, entries in buckets.items():
        if not entries:
            results[rule] = {"positions": 0,
                             "note": "aucune position trouvée dans le corpus"}
            continue

        legal_top1 = 0
        match_human = 0
        legal_mass = []

        for e in entries:
            board = chess.Board(e["fen"])
            ids = e["prefix"][-block:]
            x = torch.tensor([ids], dtype=torch.long, device=device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, _ = model(x)
            logits = logits[0, -1, :].float()
            probs = F.softmax(logits, dim=-1)

            legal_ids = [stoi[m.uci()] for m in board.legal_moves
                         if m.uci() in stoi]
            legal_mass.append(float(probs[legal_ids].sum()))

            best = itos[int(logits.argmax())]
            try:
                if chess.Move.from_uci(best) in board.legal_moves:
                    legal_top1 += 1
            except ValueError:
                pass
            if best == e["human_move"]:
                match_human += 1

        n = len(entries)
        pl, lol, hil = wilson_interval(legal_top1, n)
        pm, lom, him = wilson_interval(match_human, n)
        results[rule] = {
            "positions": n,
            "top1_legal": {"taux": round(pl, 5), "ic95_bas": round(lol, 5),
                           "ic95_haut": round(hil, 5)},
            "top1_egal_coup_humain": {"taux": round(pm, 5),
                                      "ic95_bas": round(lom, 5),
                                      "ic95_haut": round(him, 5)},
            "masse_de_probabilite_sur_coups_legaux": {
                "moyenne": round(sum(legal_mass) / n, 5),
                "min": round(min(legal_mass), 5),
                "max": round(max(legal_mass), 5),
            },
        }

    results["_methode"] = (
        "positions extraites de vraies parties du jeu de validation : pour "
        "'roque', 'en_passant' et 'promotion', les positions où l'humain a "
        "effectivement joué ce type de coup ; pour 'echec', les positions où "
        "le camp au trait est en échec et doit donc en sortir. Le modèle est "
        "interrogé sans masquage.")
    return results


# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--vocab", default="data/vocab.json")
    p.add_argument("--val-games", default="data/val_games.txt",
                   help="parties de validation en clair, une par ligne")
    p.add_argument("--device", default="cuda:0",
                   help="cuda:0 = RTX 3060, pour évaluer pendant que le 3090 entraîne")
    p.add_argument("--out", default=None)
    p.add_argument("--seed", type=int, default=1234)

    p.add_argument("--n-legal", type=int, default=5000)
    p.add_argument("--n-full-games", type=int, default=200)
    p.add_argument("--n-agreement", type=int, default=5000)
    p.add_argument("--n-per-rule", type=int, default=300)
    p.add_argument("--max-scan", type=int, default=200000)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max-plies", type=int, default=250)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--skip-full-games", action="store_true")

    args = p.parse_args()
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)

    stoi, itos, bos, eos = load_vocab(args.vocab)
    model, ck = load_model(args.ckpt, args.device)
    print(f"[modèle] {args.ckpt} | step {ck['step']:,} | "
          f"{ck['tokens_seen']:,} tokens vus")

    with open(args.val_games) as f:
        games = [l.strip() for l in f if l.strip().count(" ") >= 4]
    print(f"[données] {len(games):,} parties de validation")

    report = {
        "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "checkpoint": os.path.abspath(args.ckpt),
        "step": ck["step"],
        "tokens_vus": ck["tokens_seen"],
        "seed": args.seed,
        "parties_validation_disponibles": len(games),
    }

    t0 = time.perf_counter()
    print("\n[1/4] Taux de coups légaux en génération libre...", flush=True)
    report["legalite_generation_libre"] = legal_move_rate(
        model, stoi, itos, bos, games, args.device, args.n_legal,
        args.temperature, args.batch_size, rng)
    r = report["legalite_generation_libre"]
    print(f"      {r['taux']:.2%} "
          f"[IC95 {r['ic95_bas']:.2%} - {r['ic95_haut']:.2%}] "
          f"sur {r['positions_testees']:,} positions")

    print("\n[2/4] Accord avec le coup humain...", flush=True)
    report["accord_humain"] = human_agreement(
        model, stoi, itos, bos, games, args.device, args.n_agreement,
        args.batch_size, rng)
    r = report["accord_humain"]
    print(f"      top-1 {r['top1']['taux']:.2%} | top-5 {r['top5']['taux']:.2%}")

    print("\n[3/4] Tests par règle...", flush=True)
    report["tests_par_regle"] = rule_tests(
        model, stoi, itos, bos, games, args.device, args.n_per_rule, rng,
        args.max_scan)
    for rule in ("roque", "en_passant", "promotion", "echec"):
        d = report["tests_par_regle"].get(rule, {})
        if d.get("positions"):
            print(f"      {rule:<12} n={d['positions']:>4} | "
                  f"top1 légal {d['top1_legal']['taux']:.2%} | "
                  f"= humain {d['top1_egal_coup_humain']['taux']:.2%} | "
                  f"masse légale {d['masse_de_probabilite_sur_coups_legaux']['moyenne']:.2%}")

    if args.skip_full_games:
        report["parties_completes"] = "non mesuré (--skip-full-games)"
        print("\n[4/4] Parties complètes : ignoré")
    else:
        print("\n[4/4] Parties complètes sans coup illégal...", flush=True)
        report["parties_completes"] = full_game_rate(
            model, stoi, itos, bos, eos, args.device, args.n_full_games,
            args.temperature, args.max_plies, rng, args.batch_size)
        r = report["parties_completes"]
        print(f"      {r['taux']:.2%} "
              f"[IC95 {r['ic95_bas']:.2%} - {r['ic95_haut']:.2%}] "
              f"sur {r['parties_jouees']} parties")

    report["duree_evaluation_s"] = round(time.perf_counter() - t0, 1)

    out = args.out or os.path.join(
        "logs", f"eval_step{ck['step']}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nRapport -> {out} ({report['duree_evaluation_s']:.0f} s)")


if __name__ == "__main__":
    main()
