"""Philidor, de 1700 à 2000 : une intuition, un juge, et une recherche.

La politique (le modèle de 142 M, inchangé) propose des coups ; un juge (une
copie du même transformeur, affinée pour noter les positions sur les
évaluations Stockfish des parties Lichess annotées) dit qui gagne ; une
recherche PUCT les fait dialoguer plusieurs coups à l'avance. Ni livre
d'ouvertures, ni évaluation écrite à la main ; Stockfish ne joue pas dans le
bot, il ne sert qu'à produire les cibles du juge et à mesurer.

Les scripts se lancent depuis la racine du dépôt :
    python -m juge_recherche.<script> --help

Dans l'ordre de l'article :
    diagnostic        diagnostic_phases, diag_positions, diag_modele, diag_stockfish,
                      diag_analyse, diag_profondeur, diag_sondes_positions,
                      diag_sondes, diag_propres
    plafond           diag_plafond, engine_sfselect
    juge v1           q_candidats, etiqueter_q, juge_features, juge_train,
                      juge_verrou, engine_juge
    juge v2           extraire_eval, valeur_encoder, valeur_train, valeur_pertes,
                      engine_valeur
    recherche         recherche, recherche_verrou, engine_recherche, graphes,
                      budget_pendule
    duels             duel_lot (ouvertures appariées, parties en lot)
    distillation      iteration_etiqueter, iteration_recherche, iteration_train
                      (résultat négatif, publié comme tel)
"""
