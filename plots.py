"""Génération des graphiques PNG du projet.

Contrainte de lecture : ces images seront regardées sur mobile, dans un article
LinkedIn. D'où des polices grandes, peu de séries par graphique, des étiquettes
posées directement sur les courbes plutôt qu'une légende à décoder, et un
quadrillage discret qui ne concurrence pas les données.

Couleurs : les deux premiers slots de la palette catégorielle de référence
(bleu, orange), vérifiés avec le validateur de palette. On ne dépasse jamais
deux séries par graphique, au-delà, l'écart perceptuel entre courbes n'est
plus garanti en vision des couleurs déficiente.

Le texte porte toujours les jetons d'encre, jamais la couleur de la série :
c'est la courbe adjacente à l'étiquette qui porte l'identité, pas l'étiquette
elle-même. Une étiquette colorée serait illisible pour qui distingue mal les
teintes, alors que « validation 0.039 » posé au bout de la courbe orange reste
compréhensible sans percevoir l'orange.

Usage :
    python plots.py --run-name run1
"""

import argparse
import csv
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# --- Jetons de style -------------------------------------------------------

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a8984"
GRID = "#e6e5e1"

# Deux slots seulement : bleu et orange. Vérifié avec le validateur de palette
# (mode clair, tous appariements) : écart perceptuel 33.6 en vision normale et
# 24.7 en vision déficiente, contraste supérieur à 3:1 sur le fond pour les
# deux. Un troisième slot (aqua #1baf7a) avait été envisagé mais son contraste
# tombe à 2.74 sur ce fond clair, sous le seuil, écarté plutôt que rattrapé.
SERIES = ["#2a78d6", "#eb6834"]   # bleu, orange

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.size": 13,
    "axes.titlesize": 17,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 13,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK_SECONDARY,
    "text.color": INK_PRIMARY,
    "xtick.color": INK_SECONDARY,
    "ytick.color": INK_SECONDARY,
    "grid.color": GRID,
    "grid.linewidth": 1.0,
    "lines.linewidth": 2.0,
    "figure.dpi": 160,
})


def style_axes(ax, title, xlabel, ylabel, subtitle=None):
    # Le titre est posé au-dessus du sous-titre, tous deux en coordonnées
    # d'axes : c'est le seul moyen fiable d'éviter que les jambages du titre
    # ne mordent sur la ligne du dessous quelle que soit la taille de figure.
    if subtitle:
        ax.text(0.0, 1.13, title, transform=ax.transAxes, color=INK_PRIMARY,
                fontsize=17, fontweight="bold", va="bottom", ha="left")
        ax.text(0.0, 1.03, subtitle, transform=ax.transAxes,
                color=INK_SECONDARY, fontsize=12, va="bottom", ha="left")
    else:
        ax.text(0.0, 1.03, title, transform=ax.transAxes, color=INK_PRIMARY,
                fontsize=17, fontweight="bold", va="bottom", ha="left")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", linewidth=1.0, alpha=0.9)
    ax.set_axisbelow(True)
    # Quadrillage discret : on retire les bordures qui n'apportent rien.
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def millions(x, _):
    return f"{x/1e6:.0f}"


def save(fig, path, tight=True):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if tight:
        fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {path}")


# ---------------------------------------------------------------------------

def read_metrics_csv(path):
    train, val = [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            tok = int(row["tokens_vus"])
            if row["loss_val"]:
                val.append((tok, float(row["loss_val"])))
            elif row["loss_train"]:
                train.append((tok, float(row["loss_train"])))
    return train, val


def plot_loss(csv_path, out):
    if not os.path.exists(csv_path):
        print(f"  (loss ignorée : {csv_path} absent)")
        return
    train, val = read_metrics_csv(csv_path)
    if not train:
        print("  (loss ignorée : CSV vide)")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    xs = [t for t, _ in train]
    ys = [l for _, l in train]
    ax.plot(xs, ys, color=SERIES[0], alpha=0.85, label="entraînement")
    if val:
        vx = [t for t, _ in val]
        vy = [l for _, l in val]
        ax.plot(vx, vy, color=SERIES[1], marker="o", markersize=5,
                label="validation")

    # On réserve une marge à droite pour poser les étiquettes hors de la zone
    # tracée, plutôt que de les superposer aux courbes et à l'axe.
    xmax = max(xs)
    ax.set_xlim(min(xs) - 0.01 * xmax, xmax * 1.26)

    if val:
        # Les deux courbes finissent souvent très proches. On sépare donc les
        # étiquettes verticalement d'une fraction fixe de la hauteur du
        # graphique au lieu de les poser sur leurs valeurs respectives, qui se
        # chevaucheraient.
        span = max(max(ys), max(vy))
        ax.annotate(f"entraînement  {ys[-1]:.3f}",
                    xy=(xs[-1], ys[-1]), xytext=(10, -4),
                    textcoords="offset points", color=INK_SECONDARY,
                    fontsize=12, ha="left", va="center")
        ax.annotate(f"validation  {vy[-1]:.3f}",
                    xy=(vx[-1], vy[-1] + span * 0.06), xytext=(10, 4),
                    textcoords="offset points", color=INK_SECONDARY,
                    fontsize=12, ha="left", va="center")

    ax.xaxis.set_major_formatter(FuncFormatter(millions))
    style_axes(ax, "La loss pendant l'entraînement",
               "Tokens vus (millions)", "Loss (entropie croisée)",
               subtitle="Plus bas = le modèle prédit mieux le coup suivant")
    if val:
        ax.legend(frameon=False, loc="upper right")
    save(fig, out)


def plot_legality(eval_glob, out, final_json=None):
    """La courbe de légalité, plus le point de mesure finale.

    Les points de la courbe viennent des évaluations intermédiaires, sur 2000
    positions chacune. La mesure publiée, elle, porte sur 20 000 positions -
    dix fois plus, donc un intervalle de confiance trois fois plus serré. On
    l'ajoute distinctement plutôt que de laisser le graphique se terminer sur
    un chiffre moins précis que celui du texte : un lecteur qui compare les
    deux ne doit pas trouver d'écart inexpliqué.
    """
    files = sorted(glob.glob(eval_glob))
    points = []
    for path in files:
        with open(path) as f:
            d = json.load(f)
        leg = d.get("legalite_generation_libre")
        if isinstance(leg, dict):
            points.append((d["tokens_vus"], leg["taux"],
                           leg["ic95_bas"], leg["ic95_haut"]))
    if not points:
        print(f"  (légalité ignorée : aucun rapport d'évaluation dans {eval_glob})")
        return
    points.sort()

    fig, ax = plt.subplots(figsize=(8, 5))
    xs = [p[0] for p in points]
    ys = [p[1] * 100 for p in points]
    lo = [p[2] * 100 for p in points]
    hi = [p[3] * 100 for p in points]

    # La bande d'incertitude est dessinée sous la courbe, en transparence :
    # elle rappelle qu'un taux mesuré sur un échantillon fini est une
    # estimation et non une vérité.
    ax.fill_between(xs, lo, hi, color=SERIES[0], alpha=0.18, linewidth=0)
    ax.plot(xs, ys, color=SERIES[0], marker="o", markersize=6)

    # L'axe vertical ne part pas de zéro, et c'est justifié : sur une courbe,
    # c'est la *position* du point qui encode la valeur, pas la longueur d'une
    # forme. Tronquer l'axe d'un diagramme en barres serait malhonnête, la
    # barre mentirait sur son rapport à ses voisines, mais ici, partir de zéro
    # écraserait toute la progression de 85 à 97 % dans le dernier dixième du
    # graphique. On borne donc juste sous le minimum observé, et le premier
    # point est étiqueté pour que le niveau de départ reste explicite.
    ax.set_ylim(min(lo) - 4, 100.8)
    ax.set_xlim(min(xs) - 0.03 * max(xs), max(xs) * 1.10)

    ax.annotate(f"{ys[0]:.1f} %", xy=(xs[0], ys[0]),
                xytext=(4, -20), textcoords="offset points",
                color=INK_SECONDARY, fontsize=13, ha="left")

    # Le point de mesure finale, sur dix fois plus de positions.
    label_final = f"{ys[-1]:.1f} %"
    if final_json and os.path.exists(final_json):
        with open(final_json) as f:
            fd = json.load(f)
        fl = fd.get("legalite_generation_libre")
        if isinstance(fl, dict):
            fx, fy = fd["tokens_vus"], fl["taux"] * 100
            ax.plot([fx], [fy], marker="*", markersize=20, color=SERIES[1],
                    linestyle="none", zorder=5)
            ax.annotate(f"{fy:.2f} %\nmesure finale\nsur {fl['positions_testees']:,}"
                        .replace(",", " ") + " positions",
                        xy=(fx, fy), xytext=(-10, -46),
                        textcoords="offset points", color=INK_PRIMARY,
                        fontsize=13, fontweight="bold", ha="right")
            label_final = None
    if label_final:
        ax.annotate(label_final, xy=(xs[-1], ys[-1]),
                    xytext=(-4, -22), textcoords="offset points",
                    color=INK_PRIMARY, fontsize=16, fontweight="bold",
                    ha="right")

    ax.xaxis.set_major_formatter(FuncFormatter(millions))
    style_axes(ax, "Le modèle apprend les règles sans qu'on les lui donne",
               "Tokens vus (millions)", "Coups légaux (%)",
               subtitle="Génération libre, sans aucune contrainte · bande = IC 95 %")
    save(fig, out)


RULE_LABELS = {
    "roque": "Roque",
    "en_passant": "Prise en passant",
    "promotion": "Promotion",
    "echec": "Sortie d'échec",
}


def plot_rules(eval_glob, out):
    """Quatre règles, quatre panneaux.

    On pourrait superposer les quatre courbes sur un seul graphique, mais il
    faudrait alors quatre couleurs, et au-delà de deux séries, la palette
    validée ne garantit plus un écart perceptuel suffisant en vision des
    couleurs déficiente. La règle est alors de facetter plutôt que d'inventer
    des teintes : chaque panneau porte une seule courbe, dans la même couleur,
    et c'est le titre du panneau qui porte l'identité. Les quatre partagent le
    même axe vertical, ce qui est justement ce qui rend la comparaison lisible.
    """
    files = sorted(glob.glob(eval_glob))
    series = {r: [] for r in RULE_LABELS}
    for path in files:
        with open(path) as f:
            d = json.load(f)
        tests = d.get("tests_par_regle", {})
        for rule in RULE_LABELS:
            e = tests.get(rule)
            if isinstance(e, dict) and e.get("positions"):
                series[rule].append((d["tokens_vus"],
                                     e["top1_legal"]["taux"] * 100))
    if not any(series.values()):
        print(f"  (règles ignorées : aucun rapport dans {eval_glob})")
        return

    fig, axes = plt.subplots(2, 2, figsize=(9, 7), sharex=True, sharey=True)
    for ax, (rule, label) in zip(axes.flat, RULE_LABELS.items()):
        pts = sorted(series[rule])
        if not pts:
            ax.set_visible(False)
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, color=SERIES[0], marker="o", markersize=4)
        ax.text(0.0, 1.04, label, transform=ax.transAxes, color=INK_PRIMARY,
                fontsize=14, fontweight="bold", va="bottom")
        ax.annotate(f"{ys[-1]:.0f} %", xy=(xs[-1], ys[-1]),
                    xytext=(-4, 8), textcoords="offset points",
                    color=INK_SECONDARY, fontsize=12, ha="right")
        # Même échelle pour les quatre panneaux, bornée juste sous la plus
        # basse valeur observée. C'est tout l'intérêt du facettage : partir de
        # zéro tasserait les quatre courbes contre le haut du cadre et rendrait
        # invisible l'écart de vingt points qui est justement le sujet.
        ax.set_ylim(70, 102)
        ax.grid(True, axis="y", linewidth=1.0, alpha=0.9)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.xaxis.set_major_formatter(FuncFormatter(millions))

    fig.supxlabel("Tokens vus (millions)", color=INK_SECONDARY, fontsize=14)
    fig.supylabel("Coup le plus probable légal (%)", color=INK_SECONDARY,
                  fontsize=14)
    # Le titre est posé APRÈS tight_layout, sinon celui-ci recalcule les marges
    # et fait remonter le sous-titre dans les jambages du titre.
    fig.tight_layout()
    fig.suptitle("Le modèle apprend les exceptions avant la règle de base",
                 color=INK_PRIMARY, fontsize=17, fontweight="bold",
                 x=0.01, ha="left", y=1.07)
    fig.text(0.01, 1.015,
             "Part des positions où le coup le plus probable est légal",
             color=INK_SECONDARY, fontsize=12, ha="left")
    save(fig, out, tight=False)


def plot_elo(elo_json, out):
    # La courbe peut venir du rapport de l'échelle (instantanés les uns contre
    # les autres) ou du rapport général. On prend le premier qui la contient.
    points = []
    for path in (elo_json, os.path.join(os.path.dirname(elo_json),
                                        "elo_ladder.json")):
        if not os.path.exists(path):
            continue
        with open(path) as f:
            d = json.load(f)
        ech = d.get("echelle_instantanes", {})
        points = [p for p in ech.get("points", [])
                  if p.get("elo_diff") is not None]
        if points:
            break
    if not points:
        print(f"  (Elo ignoré : aucune courbe trouvée)")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    xs = [p["tokens_vus"] for p in points]
    ys = [p["elo_diff"] for p in points]

    # Barres d'erreur asymétriques. Un instantané qui ne gagne aucune partie a
    # une borne basse mathématiquement infinie : la formule d'Elo diverge quand
    # le score approche zéro. On ne peut donc pas lui donner une barre
    # symétrique sans inventer une précision qui n'existe pas. Ces points sont
    # tracés avec une flèche vers le bas, qui se lit « au moins aussi mauvais
    # que ça ».
    lo_arm, hi_arm, unbounded = [], [], []
    for p, y in zip(points, ys):
        b, h = p.get("elo_diff_ic95_bas"), p.get("elo_diff_ic95_haut")
        hi_arm.append((h - y) if h is not None else 0.0)
        if b is None:
            lo_arm.append(0.0)
            unbounded.append((p["tokens_vus"], y))
        else:
            lo_arm.append(y - b)
    err = [lo_arm, hi_arm]
    # La ligne du zéro est la référence : c'est le niveau du modèle final.
    # L'échelle est donc relative, et le dire explicitement évite qu'on lise
    # ces valeurs comme un Elo Lichess.
    ax.axhline(0, color=INK_MUTED, linewidth=1.2, linestyle="--")
    ax.errorbar(xs, ys, yerr=err, color=SERIES[0], marker="o", markersize=6,
                capsize=4, elinewidth=1.5)
    # Les points dont la borne basse est indéterminée reçoivent une flèche.
    for x, y in unbounded:
        ax.annotate("", xy=(x, y - 90), xytext=(x, y),
                    arrowprops=dict(arrowstyle="-|>", color=SERIES[0],
                                    linewidth=1.5, shrinkA=4, shrinkB=0))

    # Le libellé de la ligne de référence va à gauche, là où la courbe est
    # loin d'elle : posé à droite il chevaucherait l'étiquette du dernier
    # point, qui est justement collé au zéro.
    ax.annotate("niveau du modèle final", xy=(xs[0], 0), xytext=(6, 8),
                textcoords="offset points", color=INK_SECONDARY, fontsize=12,
                ha="left")
    ax.annotate(f"{ys[0]:+.0f}", xy=(xs[0], ys[0]), xytext=(8, 2),
                textcoords="offset points", color=INK_SECONDARY, fontsize=13)
    ax.annotate(f"{ys[-1]:+.0f}", xy=(xs[-1], ys[-1]), xytext=(-8, 6),
                textcoords="offset points", color=INK_PRIMARY, fontsize=14,
                fontweight="bold", ha="right")
    ax.set_xlim(min(xs) - 0.05 * max(xs), max(xs) * 1.08)
    ax.xaxis.set_major_formatter(FuncFormatter(millions))
    span = ys[-1] - ys[0]
    style_axes(ax, f"{span:.0f} points d'Elo gagnés en deux heures",
               "Tokens vus (millions)", "Écart d'Elo au modèle final",
               subtitle="Chaque instantané joue 60 parties contre le modèle final · "
                        "barres = IC 95 %, flèche = borne basse indéterminée")
    save(fig, out)


def plot_comparaison_loss(csv_a, csv_b, label_a, label_b, out):
    """Les deux entraînements sur le même axe.

    C'est le graphique qui raconte le mieux l'effet de l'échelle : même tâche,
    même code, seuls la taille du modèle et le volume de données changent. On
    trace en fonction des tokens vus et non des heures écoulées, parce que le
    temps mélangerait deux effets, le modèle plus gros voit moins de tokens
    par seconde, ce qui n'a rien à voir avec ce qu'il apprend.
    """
    series = []
    for path, label in ((csv_a, label_a), (csv_b, label_b)):
        if not os.path.exists(path):
            print(f"  (comparaison : {path} absent)")
            return
        _, val = read_metrics_csv(path)
        if not val:
            print(f"  (comparaison : pas de loss de validation dans {path})")
            return
        series.append((label, val))

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (label, val) in enumerate(series):
        xs = [t for t, _ in val]
        ys = [l for _, l in val]
        ax.plot(xs, ys, color=SERIES[i], label=label, linewidth=2)
        ax.annotate(f"{label}  {ys[-1]:.3f}", xy=(xs[-1], ys[-1]),
                    xytext=(8, 0), textcoords="offset points",
                    color=INK_SECONDARY, fontsize=12, va="center")

    xmax = max(max(t for t, _ in v) for _, v in series)
    ax.set_xlim(0, xmax * 1.32)
    lo = min(min(l for _, l in v) for _, v in series)
    ax.set_ylim(lo * 0.93, 3.2)
    ax.xaxis.set_major_formatter(FuncFormatter(millions))
    style_axes(ax, "Ce que change un modèle trois fois plus gros",
               "Tokens vus (millions)", "Loss de validation",
               subtitle="Même code, même tâche, seules la taille et la "
                        "quantité de données changent")
    ax.legend(frameon=False, loc="upper right")
    save(fig, out)


def plot_comparaison_legalite(glob_a, glob_b, label_a, label_b, out):
    """Le taux de coups légaux des deux runs, superposé."""
    series = []
    for pattern, label in ((glob_a, label_a), (glob_b, label_b)):
        pts = []
        for path in sorted(glob.glob(pattern)):
            with open(path) as f:
                d = json.load(f)
            leg = d.get("legalite_generation_libre")
            if isinstance(leg, dict):
                pts.append((d["tokens_vus"], leg["taux"] * 100))
        if pts:
            series.append((label, sorted(pts)))
    if len(series) < 2:
        print("  (comparaison légalité : il manque un des deux runs)")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (label, pts) in enumerate(series):
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, color=SERIES[i], marker="o", markersize=5, label=label)
        ax.annotate(f"{ys[-1]:.1f} %", xy=(xs[-1], ys[-1]),
                    xytext=(8, 0), textcoords="offset points",
                    color=INK_SECONDARY, fontsize=12, va="center")

    xmax = max(max(p[0] for p in pts) for _, pts in series)
    ax.set_xlim(0, xmax * 1.18)
    ymin = min(min(p[1] for p in pts) for _, pts in series)
    ax.set_ylim(ymin - 3, 100.8)
    ax.xaxis.set_major_formatter(FuncFormatter(millions))
    style_axes(ax, "Le taux de coups légaux, à deux échelles",
               "Tokens vus (millions)", "Coups légaux (%)",
               subtitle="Génération libre, sans aucune contrainte")
    ax.legend(frameon=False, loc="lower right")
    save(fig, out)


def plot_comparaison_accord(glob_a, glob_b, label_a, label_b, out):
    """L'accord avec le coup humain, pour les deux runs.

    C'est le pendant du graphique de légalité, et il raconte l'inverse : là où
    les deux courbes de légalité se superposent, celles-ci se séparent
    nettement. Les deux graphiques n'ont de sens que côte à côte : c'est leur
    contraste qui dit ce que la taille du modèle achète réellement.
    """
    series = []
    for pattern, label in ((glob_a, label_a), (glob_b, label_b)):
        pts = []
        for path in sorted(glob.glob(pattern)):
            with open(path) as f:
                d = json.load(f)
            a = d.get("accord_humain")
            if isinstance(a, dict):
                pts.append((d["tokens_vus"], a["top1"]["taux"] * 100))
        if pts:
            series.append((label, sorted(pts)))
    if len(series) < 2:
        print("  (comparaison accord : il manque un des deux runs)")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (label, pts) in enumerate(series):
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, color=SERIES[i], marker="o", markersize=5, label=label)
        ax.annotate(f"{ys[-1]:.1f} %", xy=(xs[-1], ys[-1]),
                    xytext=(8, 0), textcoords="offset points",
                    color=INK_SECONDARY, fontsize=12, va="center")

    xmax = max(max(p[0] for p in pts) for _, pts in series)
    ax.set_xlim(0, xmax * 1.18)
    ax.xaxis.set_major_formatter(FuncFormatter(millions))
    style_axes(ax, "Là où la taille du modèle change quelque chose",
               "Tokens vus (millions)", "Coup humain prédit (%)",
               subtitle="Part des positions où le coup le plus probable est "
                        "celui joué par l'humain")
    ax.legend(frameon=False, loc="lower right")
    save(fig, out)


def plot_lengths(stats_json, out):
    if not os.path.exists(stats_json):
        print(f"  (longueurs ignorées : {stats_json} absent)")
        return
    with open(stats_json) as f:
        d = json.load(f)
    hist = d.get("histogramme_longueurs")
    if not hist:
        print("  (longueurs ignorées : pas d'histogramme)")
        return

    lengths = sorted((int(k), v) for k, v in hist.items())
    xs = [k for k, _ in lengths]
    ys = [v for _, v in lengths]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(xs, ys, width=1.0, color=SERIES[0], linewidth=0)
    total = sum(ys)
    mean = sum(k * v for k, v in lengths) / total
    ax.axvline(mean, color=SERIES[1], linewidth=2, linestyle="--")
    ax.annotate(f"moyenne\n{mean:.0f} demi-coups", xy=(mean, max(ys) * 0.82),
                xytext=(10, 0), textcoords="offset points",
                color=INK_SECONDARY, fontsize=12)
    style_axes(ax, "Longueur des parties retenues",
               "Demi-coups", "Nombre de parties",
               subtitle=f"{total:,} parties après filtrage".replace(",", " "))
    save(fig, out)


def plot_throughput(csv_path, out):
    if not os.path.exists(csv_path):
        return
    xs, ys = [], []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if row["mfu"]:
                xs.append(int(row["tokens_vus"]))
                ys.append(float(row["mfu"]) * 100)
    if not xs:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(xs, ys, color=SERIES[0], alpha=0.9)
    median = sorted(ys)[len(ys) // 2]
    ax.annotate(f"médiane {median:.0f} %", xy=(xs[len(xs) // 2], median),
                xytext=(0, 14), textcoords="offset points",
                color=INK_PRIMARY, fontsize=14, fontweight="bold", ha="center")
    ax.xaxis.set_major_formatter(FuncFormatter(millions))
    ax.set_ylim(0, 100)
    style_axes(ax, "Utilisation réelle du GPU",
               "Tokens vus (millions)", "MFU (%)",
               subtitle="Part de la puissance mesurée de la carte réellement exploitée")
    save(fig, out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-name", default="run1")
    p.add_argument("--log-dir", default="logs")
    p.add_argument("--out-dir", default="figures")
    p.add_argument("--suffixe", default="",
                   help="suffixe des noms de fichiers, ex. _run2")
    p.add_argument("--eval-glob", default="eval_step*.json")
    p.add_argument("--elo-json", default="elo_report.json")
    p.add_argument("--comparer", nargs=2, metavar=("RUN_A", "RUN_B"),
                   default=None,
                   help="produit aussi les graphiques comparant deux runs")
    args = p.parse_args()
    sfx = args.suffixe

    print("Génération des graphiques...")
    plot_loss(os.path.join(args.log_dir, f"{args.run_name}_metrics.csv"),
              os.path.join(args.out_dir, f"loss{sfx}.png"))
    plot_legality(os.path.join(args.log_dir, args.eval_glob),
                  os.path.join(args.out_dir, f"legalite{sfx}.png"),
                  os.path.join(args.log_dir,
                               f"eval_final{sfx}.json" if sfx
                               else "eval_final.json"))
    plot_rules(os.path.join(args.log_dir, args.eval_glob),
               os.path.join(args.out_dir, f"regles{sfx}.png"))
    plot_elo(os.path.join(args.log_dir, args.elo_json),
             os.path.join(args.out_dir, f"elo{sfx}.png"))
    plot_lengths(os.path.join(args.log_dir, "phase1_parse_stats.json"),
                 os.path.join(args.out_dir, "longueurs_parties.png"))
    plot_throughput(os.path.join(args.log_dir, f"{args.run_name}_metrics.csv"),
                    os.path.join(args.out_dir, f"mfu{sfx}.png"))

    if args.comparer:
        a, b = args.comparer
        la = "51 M param., 1 mois" if a == "run1" else a
        lb = "142 M param., 4 mois" if b == "run2" else b
        plot_comparaison_loss(
            os.path.join(args.log_dir, f"{a}_metrics.csv"),
            os.path.join(args.log_dir, f"{b}_metrics.csv"),
            la, lb, os.path.join(args.out_dir, "comparaison_loss.png"))
        plot_comparaison_legalite(
            os.path.join(args.log_dir, "eval_step*.json"),
            os.path.join(args.log_dir, "eval_run2_step*.json"),
            la, lb, os.path.join(args.out_dir, "comparaison_legalite.png"))
        plot_comparaison_accord(
            os.path.join(args.log_dir, "eval_step*.json"),
            os.path.join(args.log_dir, "eval_run2_step*.json"),
            la, lb, os.path.join(args.out_dir, "comparaison_accord.png"))


if __name__ == "__main__":
    main()
