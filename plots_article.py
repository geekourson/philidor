"""Graphiques supplémentaires, pensés pour un article grand public.

Ceux de `plots.py` suivent la progression de l'entraînement. Ceux-ci illustrent
des constats ponctuels, ils répondent chacun à une question qu'un lecteur se
pose, et fonctionnent isolément.

Règle de rendu appliquée partout : **les diagrammes en barres partent de zéro**,
contrairement aux courbes. La différence n'est pas cosmétique. Sur une barre,
c'est la *longueur* qui encode la valeur ; tronquer l'axe fait mentir le rapport
visuel entre deux barres. Sur une courbe, c'est la *position* du point, et un
axe tronqué reste honnête tant qu'il est lisiblement gradué.

Usage :
    python plots_article.py
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plots import (SURFACE, INK_PRIMARY, INK_SECONDARY, INK_MUTED, GRID,
                   SERIES, save, style_axes)

OUT = "figures"


def barres_horizontales(ax, labels, valeurs, couleurs=None, suffixe=" %",
                        fmt="{:.1f}"):
    """Barres horizontales, valeur écrite au bout de chaque barre.

    L'horizontale est préférée quand les étiquettes sont des mots : elles se
    lisent normalement, sans rotation à 45 degrés qui oblige le lecteur à
    pencher la tête, un détail qui compte beaucoup sur mobile.
    """
    y = range(len(labels))
    couleurs = couleurs or [SERIES[0]] * len(labels)
    ax.barh(list(y), valeurs, color=couleurs, height=0.62)
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=13)
    ax.invert_yaxis()
    vmax = max(valeurs)
    for i, v in enumerate(valeurs):
        ax.text(v + vmax * 0.015, i, fmt.format(v) + suffixe,
                va="center", ha="left", fontsize=13, color=INK_PRIMARY,
                fontweight="bold")
    ax.set_xlim(0, vmax * 1.22)
    ax.grid(True, axis="x", linewidth=1.0, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="y", length=0)


# ---------------------------------------------------------------------------

def g_sicilienne():
    """Ce que le modèle propose après 1.e4 c5.

    Mesuré avec engine.py sur le checkpoint run1_best, probabilités
    renormalisées sur les seuls coups légaux.
    """
    coups = ["Cf3 : Sicilienne ouverte", "Cc3 : variante fermée",
             "c3 : Alapine", "d4 : poussée centrale",
             "Fc4 : fou italien", "f4 : attaque Grand Prix"]
    p = [54.7, 10.8, 8.9, 7.4, 5.8, 4.2]
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    barres_horizontales(ax, coups, p)
    ax.set_xlabel("Probabilité attribuée par le modèle (%)")
    # On abaisse le haut de l'axe pour dégager la place du titre et du
    # sous-titre, sinon ce dernier chevauche la première barre.
    fig.subplots_adjust(top=0.76, bottom=0.14, right=0.97)

    # Titre et sous-titre centrés sur le CONTENU (libellés y + barres), pas sur
    # l'axe. Sinon, comme les libellés débordent loin à gauche et que save()
    # recadre en bbox tight, le titre paraît décalé à droite. On calcule le
    # milieu réel au rendu.
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    gauche = min(l.get_window_extent(rend).x0 for l in ax.get_yticklabels())
    droite = ax.get_window_extent(rend).x1
    milieu = fig.transFigure.inverted().transform(((gauche + droite) / 2, 0))[0]
    fig.text(milieu, 0.95, "Il connaît la théorie des ouvertures",
             ha="center", color=INK_PRIMARY, fontsize=17, fontweight="bold")
    fig.text(milieu, 0.865,
             "Après 1.e4 c5 : les six grandes réponses à la Sicilienne, "
             "dans leur ordre de popularité réel",
             ha="center", color=INK_SECONDARY, fontsize=12)
    # tight=False : sinon fig.tight_layout() annule le subplots_adjust ci-dessus
    # et fait remonter l'axe sous le sous-titre.
    save(fig, os.path.join(OUT, "sicilienne.png"), tight=False)


def g_temperature():
    """L'effet du réglage d'échantillonnage sur le comportement observé."""
    temps = [1.0, 0.8, 0.6, 0.4, 0.2]
    sans_faute = [52.5, 68.0, 80.5, 82.5, 87.0]
    longueur = [48.2, 54.5, 64.1, 63.6, 60.5]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.4))
    for ax, vals, titre, ylab, ref in (
            (ax1, sans_faute, "Parties sans coup illégal", "%", None),
            (ax2, longueur, "Longueur des parties", "demi-coups", 71.6)):
        ax.plot(temps, vals, color=SERIES[0], marker="o", markersize=7)
        ax.invert_xaxis()
        if ref:
            ax.axhline(ref, color=SERIES[1], linewidth=2, linestyle="--")
            # Sous la ligne et non au-dessus : au-dessus elle percutait le
            # titre du panneau, la référence étant proche du haut du cadre.
            ax.text(0.98, ref - 2.2, f"vraies parties : {ref}",
                    color=INK_SECONDARY, fontsize=11, ha="left")
        ax.text(0.0, 1.05, titre, transform=ax.transAxes, color=INK_PRIMARY,
                fontsize=14, fontweight="bold", va="bottom")
        ax.set_xlabel("Température d'échantillonnage")
        ax.set_ylabel(ylab)
        ax.grid(True, axis="y", linewidth=1.0, alpha=0.9)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)

    fig.tight_layout()
    fig.suptitle("Le même modèle, mesuré à cinq réglages",
                 color=INK_PRIMARY, fontsize=17, fontweight="bold",
                 x=0.01, ha="left", y=1.16)
    fig.text(0.01, 1.045,
             "La température n'est pas un paramètre du modèle mais de la "
             "mesure. Changer d'instrument change le résultat",
             color=INK_SECONDARY, fontsize=12, ha="left")
    save(fig, os.path.join(OUT, "temperature.png"), tight=False)


def g_motifs_echec():
    """De quoi sont faites les 2 % d'erreurs."""
    try:
        d = json.load(open("logs/eval_final.json"))
        m = d["legalite_generation_libre"]["motifs_echec"]
    except Exception:
        print("  (motifs : eval_final.json absent)")
        return
    noms = {"deplacement_illegal": "Coup impossible\ndans cette position",
            "token_special": "Symbole de service\nproposé comme un coup",
            "case_depart_vide": "Déplace une pièce\nqui n'existe pas",
            "piece_adverse": "Déplace une pièce\nde l'adversaire"}
    items = sorted(m.items(), key=lambda kv: -kv[1])
    labels = [noms.get(k, k) for k, _ in items]
    vals = [v for _, v in items]

    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    barres_horizontales(ax, labels, vals, suffixe="", fmt="{:.0f}")
    ax.set_xlabel("Occurrences sur 20 000 coups générés")
    ax.text(0.0, 1.16, "Quand il se trompe, il se trompe finement",
            transform=ax.transAxes, color=INK_PRIMARY, fontsize=17,
            fontweight="bold", va="bottom")
    ax.text(0.0, 1.04,
            "Dans la moitié des cas, le coup proposé est géométriquement "
            "correct, il est juste impossible ici",
            transform=ax.transAxes, color=INK_SECONDARY, fontsize=12,
            va="bottom")
    save(fig, os.path.join(OUT, "motifs_echec.png"))


def g_filtrage():
    """Ce que le filtrage jette, à l'échelle d'un mois de Lichess."""
    try:
        d = json.load(open("logs/phase1_parse_stats.json"))
        r = d["rejets"]
        gardees = d["parties_conservees"]
    except Exception:
        print("  (filtrage : phase1_parse_stats.json absent)")
        return
    noms = {"elo_hors_bornes": "Elo hors de 1800-2600",
            "cadence_exclue": "Bullet et ultrabullet",
            "terminaison": "Fin anormale (déconnexion…)",
            "trop_courte": "Moins de 20 demi-coups",
            "trop_longue": "Plus de 300 demi-coups"}
    items = sorted(r.items(), key=lambda kv: -kv[1])
    labels = ["CONSERVÉES"] + [noms.get(k, k) for k, _ in items]
    vals = [gardees / 1e6] + [v / 1e6 for _, v in items]
    couleurs = [SERIES[1]] + [SERIES[0]] * len(items)

    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    barres_horizontales(ax, labels, vals, couleurs, suffixe=" M", fmt="{:.1f}")
    ax.set_xlabel("Millions de parties, sur un mois de Lichess")
    ax.text(0.0, 1.16, "Sur 89 millions de parties, on en garde 11",
            transform=ax.transAxes, color=INK_PRIMARY, fontsize=17,
            fontweight="bold", va="bottom")
    ax.text(0.0, 1.04,
            "Plus d'un tiers des parties jouées sur Lichess sont du bullet",
            transform=ax.transAxes, color=INK_SECONDARY, fontsize=12,
            va="bottom")
    save(fig, os.path.join(OUT, "filtrage.png"))


def g_bridage_gpu():
    """Le GPU bridé : 110 watts qui valent un facteur 2.5."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 4.2))
    for ax, vals, titre, ylab in (
            (ax1, [24.98, 62.86], "Calcul délivré", "TFLOPS bf16"),
            (ax2, [596, 1506], "Fréquence du GPU", "MHz")):
        b = ax.bar(["bridé\n170 W", "libéré\n280 W"], vals,
                   color=[INK_MUTED, SERIES[0]], width=0.55)
        for rect, v in zip(b, vals):
            ax.text(rect.get_x() + rect.get_width() / 2, v * 1.03,
                    f"{v:,.0f}".replace(",", " ") if v > 100 else f"{v:.2f}",
                    ha="center", fontsize=14, fontweight="bold",
                    color=INK_PRIMARY)
        ax.text(0.0, 1.05, titre, transform=ax.transAxes, color=INK_PRIMARY,
                fontsize=14, fontweight="bold", va="bottom")
        ax.set_ylabel(ylab)
        ax.set_ylim(0, max(vals) * 1.2)
        ax.grid(True, axis="y", linewidth=1.0, alpha=0.9)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)

    fig.tight_layout()
    fig.suptitle("110 watts qui valent un facteur 2.5",
                 color=INK_PRIMARY, fontsize=17, fontweight="bold",
                 x=0.01, ha="left", y=1.17)
    fig.text(0.01, 1.05,
             "La carte était bridée à 40 % de sa puissance, et produisait "
             "moins qu'un GPU trois fois plus petit",
             color=INK_SECONDARY, fontsize=12, ha="left")
    save(fig, os.path.join(OUT, "bridage_gpu.png"), tight=False)


def g_regles_final():
    """Les quatre règles, en barres, sur le modèle final."""
    try:
        d = json.load(open("logs/eval_final.json"))["tests_par_regle"]
    except Exception:
        print("  (règles finales : eval_final.json absent)")
        return
    noms = [("roque", "Roque"), ("en_passant", "Prise en passant"),
            ("promotion", "Promotion"), ("echec", "Sortie d'échec")]
    labels = [n for _, n in noms]
    vals = [d[k]["top1_legal"]["taux"] * 100 for k, _ in noms]
    couleurs = [SERIES[0]] * 3 + [SERIES[1]]

    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    barres_horizontales(ax, labels, vals, couleurs)
    ax.set_xlim(0, 112)
    ax.set_xlabel("Coup le plus probable légal (%), sur 500 positions")
    ax.text(0.0, 1.18, "Il apprend les exceptions avant la règle de base",
            transform=ax.transAxes, color=INK_PRIMARY, fontsize=17,
            fontweight="bold", va="bottom")
    ax.text(0.0, 1.05,
            "Sortir d'échec est la contrainte la plus fondamentale du jeu, "
            "c'est celle qu'il rate le plus",
            transform=ax.transAxes, color=INK_SECONDARY, fontsize=12,
            va="bottom")
    save(fig, os.path.join(OUT, "regles_final.png"))


def g_qwen():
    """Le chiffre central comparé, et l'effet du protocole sur la mesure.

    Quatre barres, une seule mesure : une seule couleur suffit pour nos
    modèles, une seconde distingue l'adversaire. L'axe part de zéro, comme tout
    diagramme en barres.

    Les deux barres de l'adversaire ne sont pas redondantes, elles sont le
    sujet : mesuré sur les seules positions d'ouverture il obtient 76 %, mesuré
    sur des parties menées jusqu'au mat il tombe à 36 %. Le protocole change ce
    qu'on mesure, et le montrer vaut mieux que de choisir le chiffre qui
    arrange.
    """
    import json as _j
    try:
        q_long = _j.load(open("logs/duel_qwen.json"))["legalite_premiere_tentative"]["taux"] * 100
        q_court = _j.load(open("logs/duel_qwen_run2_strict.json"))["legalite_premiere_tentative"]["taux"] * 100
        r1 = _j.load(open("logs/eval_final.json"))["legalite_generation_libre"]["taux"] * 100
        r2 = _j.load(open("logs/eval_final_run2_t10.json"))["legalite_generation_libre"]["taux"] * 100
    except Exception as e:
        print(f"  (qwen : donnees manquantes, {e})"); return

    labels = ["Mon modèle, 142 M",
              "Mon modèle, 51 M",
              "Qwen 35 Md\nsur les ouvertures",
              "Qwen 35 Md\nparties completes"]
    vals = [r2, r1, q_court, q_long]
    couleurs = [SERIES[0], SERIES[0], SERIES[1], SERIES[1]]

    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    barres_horizontales(ax, labels, vals, couleurs)
    ax.set_xlim(0, 118)
    ax.set_xlabel("Coups légaux en génération libre (%)")
    ax.text(0.0, 1.20, "250 fois plus gros, et trois fois plus d'erreurs",
            transform=ax.transAxes, color=INK_PRIMARY, fontsize=17,
            fontweight="bold", va="bottom")
    ax.text(0.0, 1.06,
            "Même question posée aux deux : un coup, sans liste des coups "
            "légaux, sans seconde chance",
            transform=ax.transAxes, color=INK_SECONDARY, fontsize=12,
            va="bottom")
    save(fig, os.path.join(OUT, "qwen_legalite.png"))


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    print("Graphiques d'article :")
    for f in (g_sicilienne, g_temperature, g_motifs_echec, g_filtrage,
              g_bridage_gpu, g_regles_final, g_qwen):
        try:
            f()
        except Exception as e:
            print(f"  ECHEC {f.__name__} : {e}")


def g_qwen_post():
    """Version pour les réseaux sociaux : trois barres, un seul protocole.

    Le graphe complet (g_qwen) montre l'effet du protocole avec quatre barres.
    Ici on veut une lecture immédiate en vignette : les deux modèles
    spécialisés contre le généraliste, tous mesurés sur des parties menées
    jusqu'au bout. Le titre porte le rapport d'erreurs, qui est le vrai écart.
    """
    import json as _j
    q = _j.load(open("logs/duel_qwen.json"))["legalite_premiere_tentative"]["taux"] * 100
    r1 = _j.load(open("logs/eval_final.json"))["legalite_generation_libre"]["taux"] * 100
    r2 = _j.load(open("logs/eval_final_run2_t10.json"))["legalite_generation_libre"]["taux"] * 100

    labels = ["Mon modèle\n142 M de paramètres",
              "Mon modèle\n51 M de paramètres",
              "Qwen3.6\n35 Md de paramètres"]
    vals = [r2, r1, q]
    couleurs = [SERIES[0], SERIES[0], SERIES[1]]

    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    barres_horizontales(ax, labels, vals, couleurs)
    ax.set_xlabel("Coups légaux au premier essai (%)")
    fig.subplots_adjust(top=0.74, bottom=0.16, right=0.97)

    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    gauche = min(l.get_window_extent(rend).x0 for l in ax.get_yticklabels())
    droite = ax.get_window_extent(rend).x1
    milieu = fig.transFigure.inverted().transform(((gauche + droite) / 2, 0))[0]
    fig.text(milieu, 0.94, "Il ne joue pas mal, il ne voit pas l'échiquier",
             ha="center", color=INK_PRIMARY, fontsize=17, fontweight="bold")
    fig.text(milieu, 0.855,
             "Même question, aucune liste de coups fournie, parties menées jusqu'au mat",
             ha="center", color=INK_SECONDARY, fontsize=12)
    save(fig, os.path.join(OUT, "qwen_post.png"), tight=False)
