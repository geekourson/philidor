#!/usr/bin/env bash
# Enchaîne les quatre duels contre Qwen, sans intervention.
#
#   run2 x 30 tentatives   (en cours au lancement de ce script)
#   run1 x 30 tentatives   -> l'adversaire affronte des positions différentes
#   run2 x strict          -> règle réelle : un coup illégal fait perdre
#   run1 x strict
#
# Pourquoi les quatre. Le taux de légalité de Qwen dépend des positions qu'on
# lui présente, donc du modèle qu'il affronte : contre un adversaire plus
# faible, les parties partent ailleurs. Mesurer avec les deux modèles sépare ce
# qui vient de Qwen de ce qui vient de la position.
#
# Les deux runs stricts sont rapides : les parties s'arrêtent au premier coup
# illégal, soit après un ou deux coups.
#
# Usage : nohup ./duels_qwen.sh > logs/duels_qwen_pipeline.log 2>&1 &

set -u
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$REPO"
PY="${PY:-$(command -v python3)}"
log() { echo "[$(date -u +%H:%M:%S)] $*"; }

# Attendre la fin du duel déjà lancé. Deux façons ratées avant celle-ci :
#
#   1. tester la présence du fichier de sortie -> il contenait encore le
#      rapport d'un run PRÉCÉDENT, le script a démarré aussitôt et deux duels
#      se sont disputé le serveur ;
#   2. comparer les dates du JSON et du log -> le log est écrit APRÈS le JSON,
#      donc la condition ne devenait jamais fausse.
#
# On attend le PID, passé en argument. C'est sans ambiguïté, contrairement à
# pgrep qui trouve aussi les commandes mentionnant simplement le motif : piège
# qui a déjà provoqué un interblocage sur ce projet.
PID_EN_COURS="${1:-}"
if [ -n "$PID_EN_COURS" ]; then
  log "attente du duel en cours (PID $PID_EN_COURS)..."
  while kill -0 "$PID_EN_COURS" 2>/dev/null; do sleep 60; done
  log "duel run2 (30 tentatives) terminé"
fi

verifier_serveur() {
  for _ in $(seq 1 30); do
    if curl -s -m 5 http://localhost:8080/health | grep -q '"ok"'; then return 0; fi
    log "  serveur llama indisponible, nouvelle tentative dans 30 s"
    sleep 30
  done
  log "ERREUR : serveur llama injoignable, arrêt"
  exit 1
}

lancer() {
  local ckpt=$1 nom=$2 sortie=$3; shift 3
  if [ -s "$sortie" ]; then log "$nom déjà fait, on saute"; return; fi
  verifier_serveur
  log "=== $nom ==="
  $PY -u qwen_match.py --ckpt "$ckpt" --device cuda:1 --games 100 \
      --out "$sortie" "$@" || log "ECHEC $nom"
}

lancer checkpoints/run1_best.pt "run1 contre Qwen, 30 tentatives" \
       logs/duel_qwen_run1.json

lancer checkpoints/run2_best.pt "run2 contre Qwen, RÈGLE RÉELLE" \
       logs/duel_qwen_run2_strict.json --strict

lancer checkpoints/run1_best.pt "run1 contre Qwen, RÈGLE RÉELLE" \
       logs/duel_qwen_run1_strict.json --strict

log "=== TOUS LES DUELS TERMINÉS ==="
for f in logs/duel_qwen.json logs/duel_qwen_run1.json \
         logs/duel_qwen_run2_strict.json logs/duel_qwen_run1_strict.json; do
  [ -s "$f" ] && $PY - "$f" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1]))
r = d["resultat_parties"]; l = d["legalite_premiere_tentative"]
ins = d.get("insistance_necessaire", {})
print(f"\n{sys.argv[1]}")
print(f"  notre modèle : +{r['victoires']} ={r['nulles']} -{r['defaites']}"
      f"   fins : {d['terminaisons']}")
print(f"  Qwen légal au 1er essai : {l['taux']:.2%} sur {l['coups_demandes']} coups")
if ins.get("tentatives_moyennes_par_coup"):
    print(f"  tentatives moyennes : {ins['tentatives_moyennes_par_coup']}")
EOF
done
