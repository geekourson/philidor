#!/usr/bin/env bash
# Étape finale : évaluations canoniques de run2, puis tous les graphiques.
#
# POLITIQUE DE TEMPÉRATURE : le point le plus important de ce script.
#
# La température n'est pas un paramètre du modèle, c'est un paramètre de la
# MESURE. Deux runs comparés à des températures différentes produisent un écart
# artificiel qu'on attribue à tort au modèle. Chaque métrique a donc une
# température imposée, et une seule :
#
#   légalité par coup, accord humain, tests par règle
#       -> température 1.0. C'est le protocole de référence, celui du 97.86 %
#          publié pour run1. C'est aussi le test le plus sévère : on
#          échantillonne dans la distribution complète du modèle.
#
#   parties complètes sans coup illégal
#       -> mesuré aux DEUX températures, et les deux publiés. À 1.0 c'est le
#          protocole de référence ; à 0.6 c'est le régime d'usage réaliste.
#          L'écart entre les deux (52.5 % contre 80.5 % sur run1) est lui-même
#          un résultat, pas un détail à cacher.
#
#   matchs et Elo
#       -> température 0, jeu déterministe, comme engine.py. La diversité des
#          parties vient des coups d'ouverture aléatoires, pas du sampling.
#
# run2.sh lance son évaluation finale à 0.6 ; on la conserve comme mesure
# secondaire et on produit ici la mesure canonique à 1.0.
set -u
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
LOGS="${LOGS:-$REPO/logs}"
mkdir -p "$LOGS"
cd "$REPO"
PY="${PY:-$(command -v python3)}"
D=data/4mois
M=$LOGS/RUN2_DONE.marker
log() { echo "[$(date -u +%H:%M:%S)] $*"; }

log "attente de la fin de run2..."
until [ -f "$M" ]; do sleep 120; done
log "run2 terminé"

# --- Évaluation canonique, température 1.0, protocole identique à run1 ------
log "=== évaluation canonique de run2 (température 1.0) ==="
$PY evaluate.py --ckpt checkpoints/run2_best.pt --device cuda:1 \
  --vocab data/vocab.json --val-games "$D/val_games.txt" \
  --n-legal 20000 --n-agreement 20000 --n-per-rule 500 --n-full-games 500 \
  --temperature 1.0 --max-scan 400000 \
  --out logs/eval_final_run2_t10.json || log "ECHEC évaluation t=1.0"

# --- Parties complètes à 0.6, régime d'usage réaliste ----------------------
# Seule cette métrique dépend fortement de la température ; les autres sont
# calculées sur l'argmax et n'en dépendent pas. On ne relance donc que ça.
log "=== parties complètes de run2 (température 0.6) ==="
$PY evaluate.py --ckpt checkpoints/run2_best.pt --device cuda:1 \
  --vocab data/vocab.json --val-games "$D/val_games.txt" \
  --n-legal 2000 --n-agreement 2000 --n-per-rule 50 --n-full-games 500 \
  --temperature 0.6 --max-scan 100000 \
  --out logs/eval_final_run2_t06.json || log "ECHEC évaluation t=0.6"

# --- Même mesure sur run1, pour que la comparaison soit symétrique ---------
log "=== parties complètes de run1 (température 0.6) ==="
$PY evaluate.py --ckpt checkpoints/run1_best.pt --device cuda:1 \
  --vocab data/vocab.json --val-games data/val_games.txt \
  --n-legal 2000 --n-agreement 2000 --n-per-rule 50 --n-full-games 500 \
  --temperature 0.6 --max-scan 100000 \
  --out logs/eval_final_run1_t06.json || log "ECHEC évaluation run1 t=0.6"

# --- Graphiques -------------------------------------------------------------
log "=== graphiques ==="
$PY plots.py --run-name run2 --suffixe _run2 \
  --eval-glob "eval_run2_step*.json" --elo-json "elo_report_run2.json" \
  || log "ECHEC graphiques run2"
$PY plots.py --run-name run1 --comparer run1 run2 || log "ECHEC comparaisons"

log "=== TOUT EST TERMINÉ ==="
ls -la figures/
