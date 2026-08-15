"""Notre modèle spécialisé contre un généraliste de 35 milliards de paramètres.

Le contraste est l'intérêt de l'exercice : d'un côté 142 M de paramètres
entraînés deux heures sur une seule tâche, de l'autre un modèle 250 fois plus
gros entraîné sur tout le web. Lequel joue le mieux aux échecs ?

### Ce qu'on mesure, et pourquoi deux choses séparément

**1. Le taux de coups légaux.** C'est la mesure directement comparable à notre
chiffre central : 97.86 % pour run1. On demande à Qwen un coup, on regarde s'il
est jouable, sans lui donner de seconde chance. Ce nombre-là est le plus
parlant de tous parce qu'il oppose les deux modèles sur exactement la même
question.

**2. Le résultat des parties.** Là, on accorde des tentatives (`--retries`).
Sans cela, presque toutes les parties se termineraient au dixième coup sur une
illégalité, et on ne mesurerait rien de la qualité du jeu. Les deux chiffres
répondent à deux questions différentes et sont publiés séparément.

### Les précautions de protocole

Un généraliste doit être interrogé correctement, sinon on mesure sa capacité à
suivre une consigne plutôt que sa force aux échecs. Trois précautions :

- on lui donne la partie en UCI **et** la liste des coups légaux n'est PAS
  fournie, la fournir reviendrait à lui offrir le masque que notre modèle
  n'a pas en génération libre ;
- on accepte sa réponse en UCI ou en notation courante, et on extrait le coup
  même s'il l'enrobe de phrases ou de raisonnement ;
- la température est celle de son service (0.6), pas 0 : c'est le régime dans
  lequel ce modèle est conçu pour fonctionner.

Usage :
    python qwen_match.py --ckpt checkpoints/run2_best.pt --games 100
"""

import argparse
import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from collections import Counter

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import chess

from elo_match import elo_with_error
from engine import ChessEngine

SYSTEM = (
    "Tu es un moteur d'échecs. On te donne les coups déjà joués d'une partie, "
    "en notation UCI (case de départ puis case d'arrivée, par exemple e2e4). "
    "Tu réponds UNIQUEMENT par le meilleur coup suivant, en notation UCI, "
    "sans aucun autre mot, sans ponctuation, sans explication."
)

UCI_RE = re.compile(r"\b([a-h][1-8][a-h][1-8][qrbn]?)\b")
THINK_RE = re.compile(r"<think>.*?</think>", re.S)


def construire_message(historique, refuses, seuil_hasard):
    """Compose la demande, en escaladant si les tentatives précédentes ont échoué.

    Trois régimes successifs, et le passage de l'un à l'autre est le coeur de
    la mesure :

    1. Première tentative : on demande simplement le meilleur coup. C'est cette
       réponse-là, et elle seule, qui compte pour le taux de légalité comparé
       à notre modèle.
    2. Tentatives suivantes : on lui dit que ses propositions précédentes
       étaient illégales et on les liste. Il dispose donc d'une information
       que notre modèle n'a jamais : c'est délibérément généreux.
    3. Au-delà du seuil : on abandonne l'exigence de qualité et on lui demande
       n'importe quel coup légal, même au hasard. S'il échoue encore là, ce
       n'est plus une question de force de jeu.
    """
    if historique:
        base = ("Coups joués : " + " ".join(historique)
                + "\n\nQuel est le meilleur coup suivant ?")
    else:
        base = "La partie commence. Quel est ton premier coup ?"

    if not refuses:
        return base

    liste = ", ".join(refuses)
    if len(refuses) < seuil_hasard:
        return (base + f"\n\nATTENTION : tu as déjà proposé {liste}. "
                f"Ces coups sont ILLÉGAUX dans cette position. "
                f"Propose un coup différent et réellement jouable.")
    return (base + f"\n\nTu as déjà proposé {len(refuses)} coups illégaux : "
            f"{liste}.\n"
            f"Oublie la qualité du coup. Donne n'importe quel coup LÉGAL, "
            f"même complètement au hasard. Le seul critère est qu'il soit "
            f"jouable dans cette position.")


def interroger(url, modele, contenu, temperature, timeout, max_tokens,
               sans_raisonnement=True):
    """Envoie la demande au serveur et renvoie le texte brut de la réponse."""

    payload = {
        "model": modele,
        "messages": [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": contenu}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    # Qwen3 est un modele de raisonnement : par defaut il produit un bloc de
    # reflexion de ~530 tokens avant de repondre, soit 7.7 s par coup, donc
    # neuf heures pour cent parties. enable_thinking=false lui fait donner la
    # meme reponse en 5 tokens et 0.4 s, soit dix-neuf fois plus vite.
    #
    # Ce n'est pas un bridage deloyal : la reponse mesuree est identique sur
    # les positions testees. Et le protocole reste plus genereux que celui de
    # notre modele, qui n'a droit qu'a un seul passage avant sans reflexion.
    if sans_raisonnement:
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    corps = json.dumps(payload).encode()

    req = urllib.request.Request(
        url, data=corps, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read())
    return d["choices"][0]["message"]["content"]


def extraire_coup(texte, board):
    """Retrouve un coup dans la réponse, en UCI ou en notation courante.

    Les modèles de raisonnement enveloppent souvent leur réponse dans un bloc
    <think>. On le retire d'abord : le coup qu'on veut est celui de la
    conclusion, pas ceux évoqués pendant la réflexion.
    """
    texte = THINK_RE.sub(" ", texte).strip()

    # On parcourt les candidats UCI en partant de la FIN : si le modèle a
    # énuméré plusieurs coups avant de conclure, sa conclusion est en dernier.
    for m in reversed(UCI_RE.findall(texte)):
        try:
            coup = chess.Move.from_uci(m)
        except ValueError:
            continue
        if coup in board.legal_moves:
            return coup, m, "uci_legal"
        return coup, m, "uci_illegal"

    # Repli sur la notation courante, au cas où il l'utilise malgré la consigne.
    for mot in reversed(texte.replace(",", " ").replace(".", " ").split()):
        mot = mot.strip("*`\"'()[]")
        try:
            coup = board.parse_san(mot)
            return coup, mot, "san_legal"
        except ValueError:
            continue

    return None, texte[:60], "illisible"


def classer_illegal(coup, board):
    """Pourquoi ce coup est-il impossible ? Même typologie que pour notre modèle."""
    p = board.piece_at(coup.from_square)
    if p is None:
        return "case_depart_vide"
    if p.color != board.turn:
        return "piece_adverse"
    return "deplacement_illegal"


def jouer_partie(eng, url, modele, notre_couleur_blanche, rng, args, stats):
    """Une partie. Renvoie le résultat du point de vue de NOTRE modèle."""
    board = chess.Board()
    historique = []

    for _ in range(args.random_plies):
        if board.is_game_over():
            break
        c = rng.choice(list(board.legal_moves))
        board.push(c)
        historique.append(c.uci())

    while not board.is_game_over(claim_draw=True) and len(historique) < args.max_plies:
        a_nous = (board.turn == chess.WHITE) == notre_couleur_blanche

        if a_nous:
            eng.set_position(board, historique)
            coup = eng.best_move(board)
        else:
            coup = None
            refuses = []
            for essai in range(args.max_attempts):
                contenu = construire_message(historique, refuses,
                                             args.seuil_hasard)
                try:
                    texte = interroger(url, modele, contenu, args.temperature,
                                       args.timeout, args.max_tokens,
                                       not args.avec_raisonnement)
                except (urllib.error.URLError, OSError, KeyError):
                    stats["erreurs_reseau"] += 1
                    time.sleep(2)
                    continue
                c, brut, genre = extraire_coup(texte, board)

                # Le taux de légalité ne compte QUE la première tentative.
                # Accorder des essais puis mesurer le meilleur fausserait la
                # comparaison avec notre modèle, qui n'en a qu'un.
                if essai == 0:
                    stats["premiere_tentative"] += 1
                    stats["p1_" + genre] += 1
                    if c is not None and c not in board.legal_moves:
                        stats["p1_type_" + classer_illegal(c, board)] += 1

                if c is not None and c in board.legal_moves:
                    coup = c
                    stats["coups_obtenus"] += 1
                    stats["total_tentatives"] += essai + 1
                    stats[f"essais_{min(essai + 1, 20)}"] += 1
                    if essai + 1 >= args.seuil_hasard:
                        stats["obtenus_apres_seuil_hasard"] += 1
                    break
                if c is not None:
                    refuses.append(brut)
                    stats["propositions_illegales"] += 1
                    if essai + 1 >= args.seuil_hasard:
                        stats["illegales_apres_seuil_hasard"] += 1
                else:
                    refuses.append("(illisible)")

            if coup is None:
                # Même avec toutes les tentatives et la consigne de jouer au
                # hasard, aucun coup jouable. On joue un coup légal à sa place
                # plutôt que d'arrêter la partie : le but est de mesurer son
                # jeu, pas de compter les forfaits.
                stats["echecs_totaux"] += 1
                stats["total_tentatives"] += args.max_attempts
                if args.forfait_si_echec:
                    return (1.0, "adversaire_illegal")
                coup = rng.choice(list(board.legal_moves))
                stats["coups_joues_a_sa_place"] += 1

        board.push(coup)
        historique.append(coup.uci())

    if len(historique) >= args.max_plies:
        return 0.5, "limite_de_coups"
    issue = board.outcome(claim_draw=True)
    if issue is None or issue.winner is None:
        return 0.5, ("indetermine" if issue is None
                     else issue.termination.name.lower())
    gagne = (issue.winner == chess.WHITE) == notre_couleur_blanche
    return (1.0 if gagne else 0.0), issue.termination.name.lower()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/run2_best.pt")
    p.add_argument("--vocab", default="data/vocab.json")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--url", default="http://localhost:8080/v1/chat/completions")
    p.add_argument("--modele", default="qwen36-35b")
    p.add_argument("--label", default="Qwen3.6-35B-A3B")
    p.add_argument("--games", type=int, default=100)
    p.add_argument("--max-attempts", type=int, default=30,
                   help="tentatives accordées à l'adversaire pour trouver un coup légal")
    p.add_argument("--seuil-hasard", type=int, default=10,
                   help="au-delà, on lui demande n'importe quel coup légal")
    p.add_argument("--forfait-si-echec", action="store_true",
                   help="perdre la partie après max-attempts, au lieu de jouer "
                        "un coup légal à sa place")
    p.add_argument("--strict", action="store_true",
                   help="RÈGLE RÉELLE : le premier coup illégal fait perdre la "
                        "partie, comme en tournoi. Équivaut à --max-attempts 1 "
                        "--forfait-si-echec.")
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--random-plies", type=int, default=2)
    p.add_argument("--max-plies", type=int, default=200)
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--avec-raisonnement", action="store_true",
                   help="laisse Qwen produire son bloc de reflexion, 19x plus lent")
    p.add_argument("--timeout", type=float, default=180)
    p.add_argument("--seed", type=int, default=99)
    p.add_argument("--out", default="logs/duel_qwen.json")
    args = p.parse_args()
    if args.strict:
        # En tournoi, un moteur qui répond un coup illégal perd immédiatement.
        # C'est la règle FIDE et celle de tout arbitre entre moteurs. Notre
        # propre moteur ne peut jamais s'y exposer : engine.py masque les coups
        # illégaux avant de choisir. Ce mode applique donc à l'adversaire
        # exactement la contrainte que subirait n'importe quel participant.
        args.max_attempts = 1
        args.forfait_si_echec = True

    # Vérification préalable : sans le serveur, autant s'arrêter tout de suite.
    sante = args.url.rsplit("/v1/", 1)[0] + "/health"
    try:
        with urllib.request.urlopen(sante, timeout=10) as r:
            r.read()
    except Exception as e:
        raise SystemExit(
            f"Serveur injoignable sur {sante} ({e}).\n"
            f"Le démarrer avec : sudo systemctl start llama.service")

    eng = ChessEngine(args.ckpt, args.vocab, args.device, temperature=0.0)
    print(f"[notre modèle] {os.path.basename(args.ckpt)}, step {eng.ckpt_step:,}, "
          f"{eng.tokens_seen/1e6:.0f} M tokens vus")
    print(f"[adversaire]   {args.label}, température {args.temperature}, "
          f"jusqu'à {args.max_attempts} tentatives, consigne de jouer "
          f"au hasard à partir de {args.seuil_hasard}")
    print(f"[protocole]    {args.games} parties, couleurs alternées, "
          f"{args.random_plies} demi-coups d'ouverture aléatoires\n")

    rng = random.Random(args.seed)
    stats = Counter()
    v = n = d_ = 0
    fins = Counter()
    t0 = time.perf_counter()

    for i in range(args.games):
        r, raison = jouer_partie(eng, args.url, args.modele, i % 2 == 0,
                                 rng, args, stats)
        fins[raison] += 1
        if r == 1.0:
            v += 1
        elif r == 0.5:
            n += 1
        else:
            d_ += 1
        if (i + 1) % 5 == 0:
            n1 = max(1, stats["premiere_tentative"])
            leg = (stats["p1_uci_legal"] + stats["p1_san_legal"]) / n1
            moy = stats["total_tentatives"] / max(1, stats["coups_obtenus"]
                                                  + stats["echecs_totaux"])
            print(f"  {i+1}/{args.games} | +{v} ={n} -{d_} | "
                  f"légaux au 1er essai : {leg:.1%} | "
                  f"{moy:.1f} tentatives/coup | "
                  f"{time.perf_counter()-t0:.0f} s", flush=True)

    from evaluate import wilson_interval
    n1 = stats["premiere_tentative"]
    legaux1 = stats["p1_uci_legal"] + stats["p1_san_legal"]
    taux, lo, hi = (wilson_interval(legaux1, n1) if n1 else (None, None, None))

    obtenus = stats["coups_obtenus"]
    moy = (stats["total_tentatives"] / max(1, obtenus + stats["echecs_totaux"]))
    distribution = {str(k): stats[f"essais_{k}"] for k in range(1, 21)
                    if stats[f"essais_{k}"]}

    rapport = {
        "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "notre_modele": {"checkpoint": os.path.abspath(args.ckpt),
                         "step": eng.ckpt_step, "tokens_vus": eng.tokens_seen},
        "adversaire": {"nom": args.label, "temperature": args.temperature,
                       "max_tentatives_par_coup": args.max_attempts,
                       "seuil_consigne_hasard": args.seuil_hasard,
                       "raisonnement_active": args.avec_raisonnement,
                       "note_raisonnement": (
                           "le bloc de reflexion de Qwen3 est desactive : il "
                           "coute 7.7 s par coup contre 0.4 s, pour une reponse "
                           "identique sur les positions testees")},
        "protocole": {
            "parties": args.games,
            "couleurs": "alternées",
            "demi_coups_aleatoires": args.random_plies,
            "note_legalite": ("le taux de légalité ne compte QUE la première "
                              "tentative de chaque coup ; accorder des essais "
                              "puis mesurer le meilleur fausserait la comparaison "
                              "avec notre modèle, qui n'en a qu'un"),
            "note_masque": ("la liste des coups légaux n'est PAS fournie à "
                            "l'adversaire, la fournir reviendrait à lui offrir "
                            "le masque que notre modèle n'a pas non plus en "
                            "génération libre")},
        "legalite_premiere_tentative": {
            "_role": ("LE chiffre comparable à notre modèle : même question, "
                      "aucune liste de coups fournie, aucune seconde chance"),
            "coups_demandes": n1,
            "legaux": legaux1,
            "taux": None if taux is None else round(taux, 5),
            "ic95_bas": None if lo is None else round(lo, 5),
            "ic95_haut": None if hi is None else round(hi, 5),
            "format_de_reponse": {k[3:]: stats[k] for k in
                                  ("p1_uci_legal", "p1_uci_illegal",
                                   "p1_san_legal", "p1_illisible")},
            "typologie_des_coups_impossibles": {
                k[9:]: stats[k] for k in
                ("p1_type_case_depart_vide", "p1_type_piece_adverse",
                 "p1_type_deplacement_illegal") if stats[k]}},

        "insistance_necessaire": {
            "_role": ("combien de relances faut-il pour obtenir un coup jouable ? "
                      "À chaque échec on lui dit que son coup était illégal et on "
                      "lui liste ses propositions refusées, une information que "
                      "notre modèle n'a jamais"),
            "coups_finalement_obtenus": obtenus,
            "tentatives_totales": stats["total_tentatives"],
            "tentatives_moyennes_par_coup": round(moy, 2),
            "distribution_tentatives": distribution,
            "propositions_illegales_totales": stats["propositions_illegales"],
            "seuil_hasard": args.seuil_hasard,
            "obtenus_apres_seuil_hasard": stats["obtenus_apres_seuil_hasard"],
            "illegales_apres_seuil_hasard": stats["illegales_apres_seuil_hasard"],
            "echecs_malgre_tout": stats["echecs_totaux"],
            "coups_joues_a_sa_place": stats["coups_joues_a_sa_place"],
            "max_attempts": args.max_attempts,
            "erreurs_reseau": stats["erreurs_reseau"]},
        "resultat_parties": elo_with_error(v, n, d_),
        "terminaisons": dict(fins),
        "duree_s": round(time.perf_counter() - t0, 1),
    }
    rapport["resultat_parties"]["_lecture"] = (
        "score et Elo du point de vue de NOTRE modèle ; un Elo positif signifie "
        "que notre modèle est plus fort")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(rapport, f, indent=2, ensure_ascii=False)

    print(f"\n=== {args.games} parties en {rapport['duree_s']/60:.0f} min ===")
    print(f"notre modèle : +{v} ={n} -{d_}  "
          f"soit {rapport['resultat_parties']['score']:.1%}")
    print(f"fins de partie : {dict(fins)}")
    if taux is not None:
        print(f"\n{args.label}, PREMIÈRE tentative : {taux:.2%} de coups légaux "
              f"[IC95 {lo:.2%} - {hi:.2%}] sur {n1:,} coups")
    print(f"  tentatives moyennes pour obtenir un coup jouable : {moy:.2f}")
    print(f"  propositions illégales au total : {stats['propositions_illegales']:,}")
    print(f"  distribution : {distribution}")
    if stats["illegales_apres_seuil_hasard"]:
        print(f"  encore {stats['illegales_apres_seuil_hasard']} illégales APRÈS "
              f"qu'on lui ait demandé un coup au hasard")
    if stats["echecs_totaux"]:
        print(f"  {stats['echecs_totaux']} coups jamais trouvés en "
              f"{args.max_attempts} tentatives")
    print(f"\nrapport -> {args.out}")


if __name__ == "__main__":
    main()
