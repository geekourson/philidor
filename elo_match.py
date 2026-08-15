"""Phase 5 : Calibration de la force de jeu.

On fait jouer le modèle contre Stockfish bridé à différents niveaux, puis on
convertit le score obtenu en différence d'Elo.

### Pourquoi pas cutechess-cli

Le plan initial prévoyait `cutechess-cli`, l'outil standard pour organiser des
tournois entre moteurs. Il n'est empaqueté ni dans Homebrew ni dans les dépôts
apt de cette machine, et le compiler demande la chaîne Qt6 complète, donc les
droits administrateur. Le juge de match tient en deux cents lignes de
python-chess, qui est déjà une dépendance du projet : on le fait à la main.
C'était aussi le critère annoncé au départ, à qualité comparable, choisir ce
qui s'explique le plus simplement.

Le seul renoncement est l'ouverture imposée : cutechess sait distribuer un
livre d'ouvertures pour diversifier les parties. On compense en jouant les
premiers coups au hasard parmi les coups légaux (voir `--random-plies`), ce qui
évite que les deux moteurs rejouent deux cents fois la même partie.

### Comment on convertit un score en Elo

La formule d'Elo dit qu'un joueur ayant D points d'écart avec son adversaire
obtient en moyenne un score de 1 / (1 + 10^(-D/400)). On l'inverse : à partir
du score mesuré, on remonte à l'écart.

    D = -400 x log10(1/score - 1)

Deux précautions. D'abord, un score de 0 ou de 100 % donne un écart infini :
on ne peut alors que donner une borne. Ensuite, le score est mesuré sur un
échantillon fini, donc il a une incertitude, qu'on propage jusqu'à l'Elo au
lieu d'annoncer un chiffre net qui n'existe pas.
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
import chess.engine

from engine import ChessEngine


# ---------------------------------------------------------------------------
# Conversion score -> Elo
# ---------------------------------------------------------------------------

def score_to_elo(score: float):
    """Écart d'Elo correspondant à un score entre 0 et 1."""
    if score <= 0.0:
        return float("-inf")
    if score >= 1.0:
        return float("inf")
    return -400.0 * math.log10(1.0 / score - 1.0)


def elo_with_error(wins: int, draws: int, losses: int, z: float = 1.96):
    """Écart d'Elo et intervalle de confiance à 95 %.

    L'incertitude vient de l'échantillonnage : sur 200 parties, un score de
    60 % ne signifie pas que la vraie valeur est 60.0 %. On estime l'écart-type
    du score à partir de la distribution observée des résultats (victoire = 1,
    nulle = 0.5, défaite = 0), puis on convertit les deux bornes du score en
    bornes d'Elo.

    Les nulles comptent double dans la précision : elles réduisent la variance
    du score, donc resserrent l'intervalle. C'est pourquoi on ne peut pas
    utiliser une simple loi binomiale ici, il faut la variance réelle des
    trois issues.
    """
    n = wins + draws + losses
    if n == 0:
        return None

    score = (wins + 0.5 * draws) / n
    # Variance des résultats individuels autour du score moyen
    variance = (wins * (1.0 - score) ** 2
                + draws * (0.5 - score) ** 2
                + losses * (0.0 - score) ** 2) / n
    stderr = math.sqrt(variance / n) if n > 0 else 0.0

    lo_score = max(0.0, score - z * stderr)
    hi_score = min(1.0, score + z * stderr)

    elo = score_to_elo(score)
    elo_lo = score_to_elo(lo_score)
    elo_hi = score_to_elo(hi_score)

    def fmt(v):
        return None if math.isinf(v) else round(v, 1)

    return {
        "parties": n,
        "victoires": wins,
        "nulles": draws,
        "defaites": losses,
        "score": round(score, 4),
        "erreur_type_score": round(stderr, 5),
        "elo_diff": fmt(elo),
        "elo_diff_ic95_bas": fmt(elo_lo),
        "elo_diff_ic95_haut": fmt(elo_hi),
        "marge_erreur": (None if (math.isinf(elo) or math.isinf(elo_lo)
                                  or math.isinf(elo_hi))
                         else round((elo_hi - elo_lo) / 2, 1)),
    }


# ---------------------------------------------------------------------------
# Une partie
# ---------------------------------------------------------------------------

def play_game(model_engine, sf, model_is_white: bool, sf_limit,
              random_plies: int, rng, max_plies: int = 400):
    """Joue une partie. Renvoie (résultat du point de vue du modèle, raison).

    Résultat : 1.0 victoire, 0.5 nulle, 0.0 défaite.
    """
    board = chess.Board()
    history: list[str] = []

    # Quelques coups au hasard pour diversifier les parties, joués par les
    # deux camps. Sans ça, deux moteurs déterministes rejouent exactement la
    # même partie à chaque fois et l'échantillon n'a aucune valeur.
    for _ in range(random_plies):
        if board.is_game_over():
            break
        move = rng.choice(list(board.legal_moves))
        board.push(move)
        history.append(move.uci())

    while not board.is_game_over(claim_draw=True) and len(history) < max_plies:
        model_turn = (board.turn == chess.WHITE) == model_is_white
        if model_turn:
            model_engine.set_position(board, history)
            move = model_engine.best_move(board)
            if move not in board.legal_moves:
                # Ne devrait jamais arriver : le masque de légalité l'interdit.
                return (0.0 if model_is_white else 0.0), "coup_illegal_du_modele"
        else:
            move = sf.play(board, sf_limit).move
            if move is None:
                break
        board.push(move)
        history.append(move.uci())

    if len(history) >= max_plies:
        return 0.5, "limite_de_coups"

    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        return 0.5, "indetermine"
    if outcome.winner is None:
        return 0.5, outcome.termination.name.lower()
    model_won = (outcome.winner == chess.WHITE) == model_is_white
    return (1.0 if model_won else 0.0), outcome.termination.name.lower()


def play_game_model_vs_model(eng_a, eng_b, a_is_white, random_plies, rng,
                             max_plies=400):
    """Deux checkpoints du même modèle s'affrontent.

    Renvoie le résultat du point de vue de `eng_a`. Sert à mesurer la
    progression : chaque instantané joue contre le modèle final, et l'écart de
    score se convertit en écart d'Elo. On obtient ainsi une courbe de force en
    fonction des tokens vus, sans avoir besoin d'un adversaire externe, donc
    sans dépendre du niveau exact auquel Stockfish est bridé.
    """
    board = chess.Board()
    history: list[str] = []

    for _ in range(random_plies):
        if board.is_game_over():
            break
        move = rng.choice(list(board.legal_moves))
        board.push(move)
        history.append(move.uci())

    while not board.is_game_over(claim_draw=True) and len(history) < max_plies:
        a_turn = (board.turn == chess.WHITE) == a_is_white
        eng = eng_a if a_turn else eng_b
        eng.set_position(board, history)
        move = eng.best_move(board)
        board.push(move)
        history.append(move.uci())

    if len(history) >= max_plies:
        return 0.5, "limite_de_coups"
    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        return 0.5, "indetermine"
    if outcome.winner is None:
        return 0.5, outcome.termination.name.lower()
    a_won = (outcome.winner == chess.WHITE) == a_is_white
    return (1.0 if a_won else 0.0), outcome.termination.name.lower()


def run_head_to_head(ckpt_a, ckpt_b, vocab, device, n_games, random_plies,
                     seed, temperature, label_a="A", label_b="B"):
    """Duel direct entre deux checkpoints, avec assez de parties pour trancher.

    Les deux modèles peuvent avoir des architectures différentes : chaque
    checkpoint porte sa propre configuration, et `ChessEngine` la lit. On peut
    donc faire s'affronter un modèle de 51 M et un de 142 M sans rien changer.

    Note sur le déterminisme : à température 0, les deux moteurs jouent toujours
    le même coup dans une position donnée. Une partie est donc entièrement
    déterminée par ses coups d'ouverture aléatoires. C'est voulu : c'est
    l'équivalent du livre d'ouvertures qu'utilisent les tournois de moteurs, et
    ça garantit que la différence mesurée vient du jeu et non du hasard
    d'échantillonnage.
    """
    eng_a = ChessEngine(ckpt_a, vocab, device, temperature)
    eng_b = ChessEngine(ckpt_b, vocab, device, temperature)
    rng = random.Random(seed)
    wins = draws = losses = 0
    reasons = Counter()
    t0 = time.perf_counter()

    for i in range(n_games):
        r, reason = play_game_model_vs_model(eng_a, eng_b, i % 2 == 0,
                                             random_plies, rng)
        reasons[reason] += 1
        if r == 1.0:
            wins += 1
        elif r == 0.5:
            draws += 1
        else:
            losses += 1
        if (i + 1) % 25 == 0:
            sc = (wins + 0.5 * draws) / (i + 1)
            print(f"    {i+1}/{n_games} | +{wins} ={draws} -{losses} | "
                  f"score {sc:.1%} | {time.perf_counter()-t0:.0f} s", flush=True)

    stats = elo_with_error(wins, draws, losses)
    stats["modele_a"] = {"label": label_a, "checkpoint": os.path.abspath(ckpt_a),
                         "step": eng_a.ckpt_step, "tokens_vus": eng_a.tokens_seen}
    stats["modele_b"] = {"label": label_b, "checkpoint": os.path.abspath(ckpt_b),
                         "step": eng_b.ckpt_step, "tokens_vus": eng_b.tokens_seen}
    stats["duree_s"] = round(time.perf_counter() - t0, 1)
    stats["terminaisons"] = dict(reasons)
    stats["_lecture"] = (f"score et Elo du point de vue de {label_a}. "
                         f"Un Elo positif signifie que {label_a} est plus fort.")
    return stats


def run_ladder(snapshots, reference_ckpt, vocab, device, n_games,
               random_plies, seed, temperature):
    """Fait jouer chaque instantané contre le modèle de référence.

    Le résultat est un écart d'Elo par instantané, relatif au modèle final.
    L'origine est donc arbitraire : c'est une courbe de progression, pas une
    échelle absolue. Le seul Elo absolu qui vaudra quelque chose sera celui
    que Lichess attribuera au bot.
    """
    ref = ChessEngine(reference_ckpt, vocab, device, temperature)
    results = []

    for step, path in snapshots:
        eng = ChessEngine(path, vocab, device, temperature)
        rng = random.Random(seed + step)
        wins = draws = losses = 0
        reasons = Counter()
        t0 = time.perf_counter()

        for i in range(n_games):
            r, reason = play_game_model_vs_model(
                eng, ref, i % 2 == 0, random_plies, rng)
            reasons[reason] += 1
            if r == 1.0:
                wins += 1
            elif r == 0.5:
                draws += 1
            else:
                losses += 1

        stats = elo_with_error(wins, draws, losses)
        stats["step"] = step
        stats["tokens_vus"] = eng.tokens_seen
        stats["duree_s"] = round(time.perf_counter() - t0, 1)
        stats["terminaisons"] = dict(reasons)
        results.append(stats)
        e = stats["elo_diff"]
        print(f"  step {step:>6} ({eng.tokens_seen/1e6:6.1f} M tokens) : "
              f"score {stats['score']:.1%} "
              f"(+{wins} ={draws} -{losses}) | "
              f"Elo {e:+.0f}" if e is not None else
              f"  step {step:>6} : score {stats['score']:.1%}, Elo non borné",
              flush=True)
        del eng

    return results


# ---------------------------------------------------------------------------
# Un match complet contre un niveau donné
# ---------------------------------------------------------------------------

def run_match(model_engine, stockfish_path, skill_level, n_games, movetime_ms,
              random_plies, seed, verbose=True):
    rng = random.Random(seed)
    sf = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    sf.configure({"Skill Level": skill_level, "Threads": 1, "Hash": 16})
    limit = chess.engine.Limit(time=movetime_ms / 1000.0)

    wins = draws = losses = 0
    reasons = Counter()
    t0 = time.perf_counter()

    try:
        for i in range(n_games):
            # On alterne les couleurs : jouer toujours blanc gonflerait le
            # score d'un avantage qui n'a rien à voir avec la force du modèle.
            model_is_white = (i % 2 == 0)
            result, reason = play_game(model_engine, sf, model_is_white, limit,
                                       random_plies, rng)
            reasons[reason] += 1
            if result == 1.0:
                wins += 1
            elif result == 0.5:
                draws += 1
            else:
                losses += 1

            if verbose and (i + 1) % 20 == 0:
                sc = (wins + 0.5 * draws) / (i + 1)
                print(f"    niveau {skill_level} : {i+1}/{n_games} parties | "
                      f"+{wins} ={draws} -{losses} | score {sc:.1%} | "
                      f"{time.perf_counter()-t0:.0f} s", flush=True)
    finally:
        sf.quit()

    stats = elo_with_error(wins, draws, losses)
    stats["skill_level"] = skill_level
    stats["movetime_ms"] = movetime_ms
    stats["random_plies"] = random_plies
    stats["duree_s"] = round(time.perf_counter() - t0, 1)
    stats["terminaisons"] = dict(reasons)
    return stats


# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--vocab", default="data/vocab.json")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--stockfish", default="stockfish")
    p.add_argument("--levels", type=int, nargs="+", default=[0, 1, 2, 3])
    p.add_argument("--games", type=int, default=200)
    p.add_argument("--movetime-ms", type=int, default=50)
    p.add_argument("--random-plies", type=int, default=4)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--out", default="logs/elo_report.json")
    p.add_argument("--ladder", action="store_true",
                   help="fait aussi jouer les instantanés contre --ckpt")
    p.add_argument("--ladder-dir", default="checkpoints")
    p.add_argument("--ladder-run", default="run1")
    p.add_argument("--ladder-games", type=int, default=60)
    p.add_argument("--skip-stockfish", action="store_true")
    p.add_argument("--vs", default=None,
                   help="checkpoint adverse pour un duel direct contre --ckpt")
    p.add_argument("--vs-games", type=int, default=300)
    p.add_argument("--label-a", default="A")
    p.add_argument("--label-b", default="B")
    args = p.parse_args()

    eng = ChessEngine(args.ckpt, args.vocab, args.device, args.temperature)
    print(f"[modèle] {args.ckpt} | step {eng.ckpt_step:,} | "
          f"{eng.tokens_seen:,} tokens vus")
    print(f"[adversaire] {args.stockfish}, {args.movetime_ms} ms par coup, "
          f"niveaux {args.levels}")
    print(f"[protocole] {args.games} parties par niveau, couleurs alternées, "
          f"{args.random_plies} demi-coups d'ouverture aléatoires\n")

    report = {
        "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "checkpoint": os.path.abspath(args.ckpt),
        "step": eng.ckpt_step,
        "tokens_vus": eng.tokens_seen,
        "protocole": {
            "parties_par_niveau": args.games,
            "movetime_ms": args.movetime_ms,
            "demi_coups_aleatoires": args.random_plies,
            "couleurs": "alternées, le modèle joue blanc une partie sur deux",
            "temperature_modele": args.temperature,
            "juge": ("python-chess, pas cutechess-cli, ce dernier n'est "
                     "empaqueté ni dans Homebrew ni dans apt sur cette machine "
                     "et sa compilation exige Qt6 et les droits administrateur"),
        },
        "matchs": [],
    }

    for level in ([] if args.skip_stockfish else args.levels):
        print(f"  == Stockfish Skill Level {level} ==", flush=True)
        stats = run_match(eng, args.stockfish, level, args.games,
                          args.movetime_ms, args.random_plies, args.seed)
        report["matchs"].append(stats)
        d = stats
        if d["elo_diff"] is None:
            print(f"    -> score {d['score']:.1%} "
                  f"(+{d['victoires']} ={d['nulles']} -{d['defaites']}) | "
                  f"écart d'Elo non borné (score extrême)\n")
        else:
            print(f"    -> score {d['score']:.1%} "
                  f"(+{d['victoires']} ={d['nulles']} -{d['defaites']}) | "
                  f"Elo {d['elo_diff']:+.0f} ± {d['marge_erreur']:.0f}\n")

    if args.vs:
        print(f"\n  == Duel direct : {args.label_a} contre {args.label_b}, "
              f"{args.vs_games} parties ==", flush=True)
        h2h = run_head_to_head(args.ckpt, args.vs, args.vocab, args.device,
                               args.vs_games, args.random_plies, args.seed,
                               args.temperature, args.label_a, args.label_b)
        report["duel_direct"] = h2h
        e, mg = h2h["elo_diff"], h2h["marge_erreur"]
        detail = (f"+{h2h['victoires']} ={h2h['nulles']} -{h2h['defaites']}")
        # La marge peut être indéfinie alors que l'Elo ne l'est pas : il suffit
        # que la borne haute de l'intervalle atteigne 100 % de score pour que
        # l'Elo correspondant diverge. Les deux valeurs se testent séparément.
        if e is None:
            print(f"    -> {args.label_a} marque {h2h['score']:.1%} ({detail}) | "
                  f"écart d'Elo non borné (score extrême)\n")
        elif mg is None:
            print(f"    -> {args.label_a} marque {h2h['score']:.1%} ({detail}) | "
                  f"Elo {e:+.0f}, marge non bornée (trop peu de parties)\n")
        else:
            print(f"    -> {args.label_a} marque {h2h['score']:.1%} ({detail}) | "
                  f"Elo {e:+.0f} ± {mg:.0f}\n")

    if args.ladder:
        import glob
        import re
        pat = os.path.join(args.ladder_dir, f"{args.ladder_run}_step*.pt")
        snaps = []
        for path in glob.glob(pat):
            m = re.search(r"_step(\d+)\.pt$", path)
            if m:
                snaps.append((int(m.group(1)), path))
        snaps.sort()
        print(f"\n  == Échelle : {len(snaps)} instantanés contre "
              f"{os.path.basename(args.ckpt)}, "
              f"{args.ladder_games} parties chacun ==", flush=True)
        ladder = run_ladder(snaps, args.ckpt, args.vocab, args.device,
                            args.ladder_games, args.random_plies, args.seed,
                            args.temperature)
        report["echelle_instantanes"] = {
            "_methode": ("chaque instantané joue contre le checkpoint de "
                         "référence passé en --ckpt, couleurs alternées, "
                         "ouvertures aléatoires. L'écart d'Elo est donc "
                         "relatif au modèle final : l'origine est arbitraire, "
                         "c'est la forme de la courbe qui a un sens."),
            "reference": os.path.abspath(args.ckpt),
            "parties_par_instantane": args.ladder_games,
            "points": ladder,
        }
        # Format directement consommable par plots.py
        report["courbe_elo_vs_tokens"] = [
            {"tokens_vus": p["tokens_vus"],
             "elo": p["elo_diff"],
             "marge_erreur": p["marge_erreur"]}
            for p in ladder
            if p["elo_diff"] is not None and p["marge_erreur"] is not None
        ]

    report["note_elo_absolu"] = (
        "Les chiffres ci-dessus sont des écarts d'Elo *relatifs* à Stockfish "
        "au niveau et à la cadence indiqués, mesurés sur cette machine. Les "
        "convertir en Elo absolu sur l'échelle Lichess demanderait un point "
        "d'ancrage que ce projet n'a pas vérifié lui-même. Le seul Elo absolu "
        "qui vaudra sera celui que Lichess attribuera au bot après ses "
        "premières parties classées, à la phase 6.")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Rapport -> {args.out}")


if __name__ == "__main__":
    main()
