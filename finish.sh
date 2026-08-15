#!/usr/bin/env bash
# Enchaîne tout ce qui suit l'entraînement, sans intervention.
#
#   1. attend la fin du run
#   2. évaluation finale sur gros échantillons (le chiffre publiable)
#   3. échelle Elo des instantanés entre eux
#   4. matchs contre Stockfish, du plus faible au plus fort
#   5. regénération des graphiques
#
# Chaque étape écrit son log et son JSON. Une étape qui échoue n'empêche pas
# les suivantes : mieux vaut trois résultats sur quatre qu'un arrêt complet
# après la première erreur.
#
# Usage : nohup ./finish.sh > logs/finish.log 2>&1 &

set -u
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$REPO"
PY="${PY:-$(command -v python3)}"
SF="${STOCKFISH:-$(command -v stockfish)}"
CKPT=checkpoints/run1_best.pt

log() { echo "[$(date -u +%H:%M:%S)] $*"; }

log "attente de la fin de l'entraînement..."
while pgrep -f "train.py --run-name run1" > /dev/null; do sleep 30; done
log "entraînement terminé"

# On n'attend PAS que le watcher se termine de lui-même : il patiente 45
# minutes sans nouvel instantané avant de s'arrêter, ce qui retarderait tout
# d'autant. Et cette attente serait de toute façon inutile, le watcher
# travaille sur le 3060 tandis que tout ce qui suit tourne sur le 3090, donc
# les deux ne se disputent rien.
# On laisse simplement une courte grâce pour que le dernier instantané soit
# évalué et rejoigne la courbe de progression, puis on arrête le watcher.
log "grâce de 3 min pour l'évaluation du dernier instantané..."
sleep 180
if pgrep -f "eval_watcher.py" > /dev/null; then
  log "arrêt du watcher"
  pkill -f "eval_watcher.py"
fi

if [ ! -f "$CKPT" ]; then
  log "ERREUR : $CKPT introuvable, arrêt"
  exit 1
fi

# --- 2. Évaluation finale -------------------------------------------------
# Échantillons dix fois plus grands que ceux des évaluations intermédiaires :
# ici on ne trace pas une courbe, on publie un chiffre. L'intervalle de
# confiance se resserre comme la racine du nombre d'observations, donc passer
# de 2 000 à 20 000 positions divise sa largeur par un peu plus de trois.
log "=== évaluation finale ==="
$PY evaluate.py \
  --ckpt "$CKPT" --device cuda:1 \
  --n-legal 20000 --n-agreement 20000 \
  --n-per-rule 500 --n-full-games 500 \
  --max-scan 400000 \
  --out logs/eval_final.json 2>&1 || log "ECHEC de l'évaluation finale"

# --- 3. Échelle Elo des instantanés ---------------------------------------
log "=== échelle Elo des instantanés ==="
$PY elo_match.py \
  --ckpt "$CKPT" --device cuda:1 \
  --skip-stockfish --ladder --ladder-games 60 \
  --out logs/elo_ladder.json 2>&1 || log "ECHEC de l'échelle Elo"

# --- 4. Matchs contre Stockfish -------------------------------------------
# Les niveaux sont joués dans l'ordre croissant : si le temps manque, on aura
# au moins les niveaux faibles, qui sont ceux où le modèle a une chance de
# marquer des points et donc où l'Elo est mesurable plutôt que borné.
log "=== matchs contre Stockfish ==="
$PY elo_match.py \
  --ckpt "$CKPT" --device cuda:1 \
  --stockfish "$SF" \
  --levels 0 1 2 3 --games 200 --movetime-ms 50 \
  --out logs/elo_report.json 2>&1 || log "ECHEC des matchs Stockfish"

# --- 5. Graphiques ---------------------------------------------------------
log "=== graphiques ==="
$PY plots.py --run-name run1 2>&1 || log "ECHEC des graphiques"

log "=== TOUT EST TERMINÉ ==="
