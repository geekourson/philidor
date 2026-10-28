# Raw outputs · Sorties brutes

Every measurement quoted in the README section *From 1700 to 2000* / *De 1700 à 2000*, as written by the scripts. Duels: paired openings, colours alternated, 95 % intervals.

Chaque mesure citée dans la section *De 1700 à 2000* du README, telle qu'écrite par les scripts. Duels : ouvertures appariées, couleurs alternées, intervalles à 95 %.

| File · Fichier | Measurement · Mesure | Result · Résultat |
|---|---|---|
| `mesure1.json` | Blunder rate of the policy, best move in the top-k · Taux de gaffes de la politique, meilleur coup dans le top-k | 18.32 % |
| `plafond.json` | Ceiling, offline: Stockfish depth 1 to 12 choosing in the top-5 · Plafond hors ligne | 14.94 % → 5.14 % |
| `e0_clone.json`, `e0_p1.json`, `e0_p4.json`, `e0_p12.json` | Ceiling in games: Stockfish depth 1 / 4 / 12 choosing in the top-5 vs the policy, and clone control · Plafond en partie | +126.8 / +329.1 / +599.4 |
| `e1_clone.json`, `e1_juge.json` | Judge v1 (small head) vs the policy, and clone control · Juge v1 contre la politique | +51.8 |
| `e1b_clone.json`, `e1b_valeur.json` | Judge v2 (full copy, 1 month) vs the policy, and clone control · Juge v2 contre la politique | +129.9 |
| `e1b_contre_juge.json` | Judge v2 vs judge v1, head to head · Juge v2 contre juge v1 | +54.3 |
| `valeur_pertes.json` | Loss profile of every selector on the 5,000 positions · Profil de pertes de chaque sélecteur | |
| `lot_valeur_argmax.json` | Batched duels, validation against the one-game-at-a-time duel · Validation des duels en lot | +118.1 (400 games) |
| `diag.json` | Search, 16 / 64 / 256 simulations, 5,000 positions (1-month judge) · Recherche | 13.74 / 11.64 / 8.10 % |
| `lot_recherche_valeur.json` | Search 64 vs policy + judge · Recherche 64 contre politique + juge | +381.7 |
| `lot_clone.json` | Batched clone control, search vs itself · Témoin clone en lot | +10.4 [−52.7 ; +74.3] |
| `lot_juge4mois_juge1mois_valeur.json`, `lot_juge4mois_juge1mois_recherche.json`, `diag_v2.json` | Judge trained on 4 months vs 1 month, without and with search · Juge 4 mois contre 1 mois | +33.1 / +37.0, 10.98 % |
| `lot_256_64.json` | Search 256 vs 64 simulations · Recherche 256 contre 64 | +415.6 |
| `diag_1024.json`, `diag_1024_lot32.json` | 256 vs 1024 simulations on 1,000 positions, and batches of 32 · 256 contre 1024, et lots de 32 | 9.50 / 6.90 %, 8.20 % |
| `lot_1024_256.json` | Search 1024 vs 256 simulations · Recherche 1024 contre 256 | +326.4 |
| `diag_4096.json` | 1024 vs 4096 simulations on 400 positions, move by move · 1024 contre 4096, position par position | 4.75 / 4.75 % |
| `lot_v3_v2_valeur.json` | Judge v3 vs v2, policy + judge without search, 2,000 games · Juge v3 contre v2, sans recherche | +20.7 |
| `lot_v3_v2_recherche.json`, `lot_v3_v2_recherche_b.json` | Judge v3 vs v2, search 64, 2 × 1,000 games (seeds 1234, 4321) · Juge v3 contre v2, recherche 64 | +16.0 / +17.4, cumulated +16.7 [+2.7 ; +30.8] |
| `lot_v4_v3.json` | Judge v4 (6 months) vs v3, search 64 · Juge v4 contre v3 | +16.3 |
| `largeur.json` | Tree width k 5 / 8 / 12 at 1024 simulations, 1,000 positions, choices per position · Largeur de l'arbre | 6.70 / 5.30 / 5.20 % |
| `lot_k12_k5.json` | Search 1024, 12 vs 5 moves per node · Recherche 1024, 12 contre 5 coups par nœud | +111.4 |
| `lot_v5_v4.json` | Judge v5 (search leaves + bot games) vs v4, search 64 · Juge v5 contre v4 | +6.6, not significant |
| `profondeur12.json` | 2048 and 4096 simulations at width 12, 1,000 positions, choices per position · 2048 et 4096 simulations à largeur 12 | 5.00 / 4.00 % (1024: 5.20 %) |
| `cpuct12.json` | cpuct 0.05 / 0.1 / 0.2 at 1024 simulations, width 12, bot's positions · Réglage de cpuct | 7.70 / 7.00 / 7.50 % |
| `lot_reprise.json` | Search 256, width 12, with vs without tree reuse, 400 games · Avec contre sans reprise de l'arbre | +122.0 |
| `lot_4096_1024_k12.json` | Search 4096 vs 1024 simulations, width 12, 200 games · Recherche 4096 contre 1024, largeur 12 | +202.6 |
