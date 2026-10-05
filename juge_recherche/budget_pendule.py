"""Budget de recherche par coup : viser ~80 % de la pendule, l'effort au milieu
de partie, et dégrader jusqu'à UNE passe quand le temps manque.

But : utiliser ~80 % de la pendule (le premier budget n'en utilisait que 10 à
20 %), et en fin de pendule descendre jusqu'à une seule passe.

Mesure (26/09, 6,26 M positions de B1) : le juge contredit l'intuition dans
11 % des positions d'ouverture (+0,56 % de chances de gain) contre 21-22 % au
milieu de partie (+1,6-1,8 %), et le gain retombe en finale longue. D'où un
poids par moment de la partie.

Temps visé pour ce coup :
    ((temps restant - réserve) / coups restants estimés) x poids(demi-coup) x ECHELLE
    + 0,8 x incrément - latence, divisé par les demandes en cours.
Coûts mesurés sur la 3060 à côté du bot (graphes CUDA) : 0,14 s + 2,9 ms par
simulation, 60 ms pour le juge seul (2 passes), ~30 ms pour l'intuition seule.
Le nombre de simulations se déduit du temps visé :
    >= 16 simulations possibles -> recherche (plafond sims_max)
    sinon, >= 0,15 s            -> 0 : politique + juge (2 passes)
    sinon                       -> -1 : politique seule (1 passe)
"""

# Coûts mesurés sur la 3060 à côté du bot, graphes CUDA, lots de 8 (26/09 soir) :
# 16 sim. 0,17 s ; 64 : 0,33 s ; 256 : 0,88 s ; 1024 : 3,12 s.
COUT_FIXE = 0.14      # s
COUT_SIM = 0.0029     # s par simulation
LATENCE = 0.1         # s par coup : réseau, lichess-bot (Lichess en compense une partie)
ECHELLE = 1.0         # calibré par simulation : ~80 % a 45 coups, jamais au temps a 120
PLANCHER = 20         # coups restants estimés, jamais moins
RESERVE_COUPS = 40    # on garde toujours de quoi jouer 40 coups en une passe


def poids(ply):
    if ply < 16:
        return 0.35       # ouverture : l'intuition suffit presque toujours
    if ply < 32:
        return 0.8
    if ply < 80:
        return 1.3        # milieu de partie : là où réfléchir rapporte le plus
    return 0.9            # finale


def coups_restants(ply):
    """Coups restants estimés pour le bot (parties de ~42 coups en moyenne)."""
    return max(float(PLANCHER), 40.0 - 0.6 * (ply // 2))


def depuis_temps(cible_s, sims_max):
    if cible_s < 0.15:
        return -1
    s = int((cible_s - COUT_FIXE) / COUT_SIM)
    if s < 16:
        return 0
    return min(sims_max, s - s % 8)


def temps_vise(t_ms, inc_ms, ply):
    """Secondes visées pour ce coup (avant partage entre parties simultanées)."""
    t, inc = t_ms / 1000, (inc_ms or 0) / 1000
    reserve = (0.03 + LATENCE) * RESERVE_COUPS
    return max(0.0, t - reserve) / coups_restants(ply) * poids(ply) * ECHELLE + 0.8 * inc - LATENCE


def patience(t_ms, inc_ms, ply):
    """Attente tolérée par une partie derrière une recherche en cours : un quart
    de son temps visé, au moins 0,3 s. Un coup de bullet interrompt vite une
    longue recherche ; une autre partie rapide la laisse travailler."""
    if t_ms is None:
        return 5.0
    return max(0.3, 0.25 * temps_vise(t_ms, inc_ms, ply))


def simulations(t_ms, inc_ms, ply, sims_max, en_cours=1):
    """t_ms : temps restant du camp au trait (None = sans pendule)."""
    if t_ms is None:
        return sims_max
    return depuis_temps(temps_vise(t_ms, inc_ms, ply) / max(1, en_cours), sims_max)


def palier(alloue_ms, sims_max):
    """Cas `go movetime` : tout le temps alloué, moins la latence."""
    return depuis_temps(alloue_ms / 1000 - LATENCE, sims_max)


def cout(s):
    return 0.03 if s < 0 else COUT_FIXE + COUT_SIM * max(0, s)


if __name__ == "__main__":
    import sys
    ech = float(sys.argv[1]) if len(sys.argv) > 1 else ECHELLE
    ECHELLE = ech
    print(f"ECHELLE {ech} | par cadence : % de la pendule utilise (parties de 30 / 45 / 70 / 100 coups), "
          f"simulations typiques coups 1-8 / 9-16 / 17-40 / 41+, fin")
    for nom, base, inc in (("bullet 1+0", 60, 0), ("blitz 3+0", 180, 0), ("blitz 3+2", 180, 2),
                           ("rapide 10+0", 600, 0), ("rapide 10+5", 600, 5), ("classique 30+0", 1800, 0)):
        lignes = []
        for n in (30, 45, 70, 100):
            t = base; prof = []; perdu = None
            for m in range(n):
                s = simulations(t * 1000, inc * 1000, 2 * m, 4096)
                t -= cout(s) + LATENCE; prof.append(s)
                if t <= 0:
                    perdu = m; break
                t += inc
            util = 1 - t / (base + inc * n)
            lignes.append((n, util, perdu, prof))
        n, u, p, prof = lignes[1]
        med = lambda xs: sorted(xs)[len(xs) // 2] if xs else None
        tranche = [med(prof[0:8]), med(prof[8:16]), med(prof[16:40]), med(prof[40:])]
        fin100 = lignes[3][3][-5:]
        print(f"  {nom:<15} " + " / ".join(f"{u:>4.0%}" + ("!" if p is not None else "") for _, u, p, _ in lignes)
              + f"   45 coups : {tranche}   fin d'une partie de 100 coups : {fin100}")
