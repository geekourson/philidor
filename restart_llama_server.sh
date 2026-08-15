#!/usr/bin/env bash
# Relance le serveur LLM de Billy exactement comme il tournait avant que ce
# projet ne réquisitionne les GPU.
#
# Ligne de commande capturée le 2026-08-04 à 13:29 UTC depuis `ps aux` sur le
# PID 1039, qui tournait alors depuis 21 h 44 min (démarré le 2026-08-03).
# Le service répondait {"status":"ok"} sur http://localhost:8080/health.
#
# Occupation VRAM constatée avant arrêt :
#   RTX 3060 (bus 06:00.0) : 8127 MiB
#   RTX 3090 (bus 07:00.0) : 17542 MiB
#
# Usage : ./restart_llama_server.sh

set -euo pipefail

LOG="${LOG:-$HOME/llama-server.log}"
LLAMA_SERVER="${LLAMA_SERVER:-$HOME/llama.cpp/build/bin/llama-server}"

nohup "$LLAMA_SERVER" \
  -hf unsloth/Qwen3.6-35B-A3B-GGUF:UD-Q4_K_XL \
  --alias qwen36-35b \
  --jinja \
  -ngl 99 \
  -fa on \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  -c 131072 \
  --temp 0.6 \
  --top-p 0.95 \
  --top-k 20 \
  --min-p 0 \
  # 127.0.0.1 : le serveur n'écoute QUE en local. L'API de llama-server n'a
  # aucune authentification ; l'exposer sur 0.0.0.0 la rendrait accessible à
  # toute personne sur le réseau. À ne changer qu'en connaissance de cause.
  --host 127.0.0.1 \
  --port 8080 \
  > "$LOG" 2>&1 &

echo "llama-server relancé, PID $!, logs dans $LOG"
echo "Vérifier avec : curl -s http://localhost:8080/health"
