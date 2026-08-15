#!/usr/bin/env bash
# Second entraînement : modèle plus gros sur quatre mois de données.
#
#   modèle  : 20 couches, d=768, 12 têtes, MLP 2048 -> 141 589 248 paramètres
#   données : dumps Lichess 2026-04 à 2026-07 -> ~3.06 G tokens
#   ratio   : 21.6 tokens par paramètre, soit l'optimum de Chinchilla
#
# Le filtre Elo reste 1800-2600. Le relever diviserait le corpus par 2.6
# (mesuré sur 60 000 parties), or un modèle plus gros a besoin de PLUS de
# données, pas moins. Les deux pistes se contredisent frontalement.
#
# Batch 80 et non 96 : à batch 96 le pic mémoire atteint 22.8 Go sur 24, trop
# juste pour un run de vingt heures, alors que le débit est identique
# (34 598 contre 34 884 tokens/s). On ne paie donc rien pour cette marge.
#
# Usage : nohup ./run2.sh > logs/run2_pipeline.log 2>&1 &

set -u
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$REPO"
PY="${PY:-$(command -v python3)}"
D=data
MOIS="2026-06 2026-05 2026-04"      # 2026-07 est déjà parsé
log() { echo "[$(date -u +%H:%M:%S)] $*"; }

# --- 1. Attendre les téléchargements ---------------------------------------
# On teste la TAILLE des fichiers, pas la présence d'un processus. Un pgrep
# sur un motif trouve aussi les commandes qui ne font que mentionner ce motif
# : c'est exactement ce qui a provoqué un interblocage plus tôt aujourd'hui.
log "attente des téléchargements..."
for m in $MOIS; do
  f="$D/lichess_db_standard_rated_${m}.pgn.zst"
  attendu=$(curl -sI "https://database.lichess.org/standard/lichess_db_standard_rated_${m}.pgn.zst" \
            | grep -i content-length | tr -d '\r' | awk '{print $2}')
  [ -z "$attendu" ] && { log "ERREUR : taille de $m introuvable"; exit 1; }
  while [ "$(stat -c %s "$f" 2>/dev/null || echo 0)" -lt "$attendu" ]; do sleep 30; done
  log "  $m complet ($(numfmt --to=iec "$attendu"))"
done

# --- 2. Parser les nouveaux mois -------------------------------------------
for m in $MOIS; do
  out="$D/games_uci_${m}.txt"
  if [ -s "$out" ]; then log "$m déjà parsé, on saute"; continue; fi
  log "=== parsing $m ==="
  $PY prepare_data.py parse \
    --input "$D/lichess_db_standard_rated_${m}.pgn.zst" \
    --workers 10 --target-tokens 0 --max-games 0 \
    --out "$out" --stats "logs/phase1_parse_${m}.json" \
    || { log "ECHEC parsing $m"; exit 1; }
done

# --- 3. Corpus combiné ------------------------------------------------------
log "=== fusion des quatre mois ==="
COMBINED="$D/games_uci_4mois.txt"
cat "$D/games_uci.txt" "$D/games_uci_2026-06.txt" \
    "$D/games_uci_2026-05.txt" "$D/games_uci_2026-04.txt" > "$COMBINED"
log "  $(wc -l < "$COMBINED") parties au total"

# --- 4. Encodage ------------------------------------------------------------
# Sautable comme le parsing : sur un relancement après incident, refaire huit
# minutes d'encodage pour rien serait absurde.
mkdir -p "$D/4mois"
if [ -s "$D/4mois/train.bin" ] && [ -s "$D/4mois/encode_stats.json" ]; then
  log "=== encodage déjà fait, on saute ==="
else
  log "=== encodage ==="
  $PY prepare_data.py encode \
    --games "$COMBINED" --vocab "$D/vocab.json" --outdir "$D/4mois" \
    || { log "ECHEC encodage"; exit 1; }
fi
cp -n "$D/vocab.json" "$D/4mois/vocab.json" 2>/dev/null || true

TOKENS=$($PY -c "import json;print(json.load(open('$D/4mois/encode_stats.json'))['tokens_train'])")
log "  train.bin : $TOKENS tokens"

# --- 5. Entraînement --------------------------------------------------------
# batch 80 x accumulation 2 = 40 960 tokens par step.
STEPS=$(( TOKENS / 40960 ))
log "=== entraînement : $STEPS steps pour une époque ==="

$PY train.py \
  --run-name run2 --device cuda:1 \
  --data-dir "$D/4mois" \
  --n-layer 20 --n-embd 768 --n-head 12 --mlp-hidden 2048 \
  --batch-size 80 --grad-accum 2 \
  --max-steps "$STEPS" --warmup-steps 2000 \
  --eval-interval 1000 --snapshot-every 5000 \
  || { log "ECHEC entraînement"; exit 1; }

# --- 6. Évaluation et Elo ---------------------------------------------------
# Température 0.6 et non 1.0 : le balayage fait sur run1 a montré qu'à
# température 1 le modèle échantillonne <eos> par accident et coupe les
# parties trop tôt, ce qui fausse le taux de parties complètes.
log "=== évaluation finale ==="
$PY evaluate.py --ckpt checkpoints/run2_best.pt --device cuda:1 \
  --vocab "$D/vocab.json" --val-games "$D/4mois/val_games.txt" \
  --n-legal 20000 --n-agreement 20000 --n-per-rule 500 --n-full-games 500 \
  --temperature 0.6 --max-scan 400000 \
  --out logs/eval_final_run2.json || log "ECHEC évaluation"

# Le duel qui intéresse vraiment : les deux modèles finaux face à face, avec
# assez de parties pour que la différence soit tranchée plutôt que suggérée.
# 400 parties donnent une marge d'environ ±35 Elo, contre ±90 pour 60 parties.
log "=== DUEL : run2 final contre run1 final, 400 parties ==="
$PY elo_match.py --ckpt checkpoints/run2_best.pt --device cuda:1 \
  --vocab "$D/vocab.json" --skip-stockfish \
  --vs checkpoints/run1_best.pt --vs-games 400 \
  --label-a "run2 (142 M, 4 mois)" --label-b "run1 (51 M, 1 mois)" \
  --out logs/duel_run2_vs_run1.json || log "ECHEC duel"

log "=== run2 contre les instantanés de run1 ==="
$PY elo_match.py --ckpt checkpoints/run2_best.pt --device cuda:1 \
  --vocab "$D/vocab.json" --skip-stockfish --ladder \
  --ladder-dir checkpoints --ladder-run run1 --ladder-games 100 \
  --out logs/elo_run2_vs_run1.json || log "ECHEC comparaison run1"

log "=== run2 contre Stockfish ==="
$PY elo_match.py --ckpt checkpoints/run2_best.pt --device cuda:1 \
  --vocab "$D/vocab.json" \
  --stockfish "${STOCKFISH:-$(command -v stockfish)}" \
  --levels 0 1 2 3 4 --games 200 --movetime-ms 50 \
  --out logs/elo_report_run2.json || log "ECHEC Stockfish"

log "=== RUN2 TERMINÉ ==="
