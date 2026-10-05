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
