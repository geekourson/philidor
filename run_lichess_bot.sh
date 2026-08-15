#!/usr/bin/env bash
# Watchdog : gère LE serveur d'inférence + lichess-bot, les maintient en vie, et
# garantit qu'aucun processus ne survit à un arrêt. Lancé une seule fois :
#
#   cd /chemin/vers/chess-model
#   setsid ./run_lichess_bot.sh > $LOGS/watchdog.log 2>&1 < /dev/null &
#   disown
#
# ARCHITECTURE (voir INFERENCE_SERVER.md). Le modèle est sans état : au lieu
# d'une copie par partie, un unique serveur (infer_server.py) le charge une fois,
# et chaque partie utilise un client UCI léger (engine_client.py) qui relaie les
# coups vers ce serveur par une socket Unix. La concurrence n'est donc plus
# limitée par la mémoire.
#
# ARRÊT PROPRE : kill le PID de CE script (dans watchdog.pid). Le trap tue le
# bot, le serveur et tout worker restant, puis sort.
#   kill "$(cat $LOGS/watchdog.pid)"
#
# POURQUOI tout ce soin sur les processus. lichess-bot lance des workers
# `multiprocessing` ; tués à moitié, ils survivent (PPID=1) et gardent le flux
# Lichess ouvert -> compte bloqué en « 429 » (un flux par compte). D'où : bot
# lancé via `setsid` (chef de session, kill -- -PGID fiable) et fonction
# nettoyer() de sécurité au démarrage ET à l'arrêt. Historique : LICHESS_BOT.md.

set -u
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
LOGS="${LOGS:-$REPO/logs}"
mkdir -p "$LOGS"
BOT_DIR="${BOT_DIR:-$REPO/lichess-bot}"
PY="${PY:-$(command -v python3)}"
LOG=$LOGS/lichess_bot.log
SRV_LOG=$LOGS/infer_server.log
PIDFILE=$LOGS/watchdog.pid
SOCK=$LOGS/infer.sock
CKPT="$REPO/checkpoints/run2_best.pt"
VOCAB="$REPO/data/vocab.json"
DEVICE=cuda:0
RESTART_DELAY=120

srv_pid=""
child=""

# Tue tout reste : serveur, bots (+ leur session), workers orphelins (PPID=1).
nettoyer() {
  for p in $(pgrep -f 'infer_server\.py' 2>/dev/null); do kill -KILL "$p" 2>/dev/null; done
  for b in $(pgrep -f 'lichess-bot\.py' 2>/dev/null); do
    local sid; sid=$(ps -o sid= -p "$b" 2>/dev/null | tr -d ' ')
    [ -n "$sid" ] && kill -TERM -- -"$sid" 2>/dev/null
  done
  sleep 1
  for b in $(pgrep -f 'lichess-bot\.py' 2>/dev/null); do
    local sid; sid=$(ps -o sid= -p "$b" 2>/dev/null | tr -d ' ')
    [ -n "$sid" ] && kill -KILL -- -"$sid" 2>/dev/null
    kill -KILL "$b" 2>/dev/null
  done
  for pid in $(ls /proc 2>/dev/null | grep -E '^[0-9]+$'); do
    local comm cmd ppid
    comm=$(cat /proc/$pid/comm 2>/dev/null) || continue
    case "$comm" in python*) ;; *) continue ;; esac
    cmd=$(tr '\0' ' ' < /proc/$pid/cmdline 2>/dev/null)
    case "$cmd" in *multiprocessing*) ;; *) continue ;; esac
    ppid=$(awk '/^PPid:/{print $2}' /proc/$pid/status 2>/dev/null)
    [ "$ppid" = "1" ] && kill -KILL "$pid" 2>/dev/null
  done
}

# (Re)démarre le serveur d'inférence et attend qu'il soit prêt (modèle chargé).
demarrer_serveur() {
  for p in $(pgrep -f 'infer_server\.py' 2>/dev/null); do kill -KILL "$p" 2>/dev/null; done
  rm -f "$SOCK" "$SOCK.ready"
  echo "[watchdog $(date -u +%H:%M:%S)] démarrage du serveur d'inférence ($DEVICE)"
  ( cd "$REPO" && setsid "$PY" infer_server.py --ckpt "$CKPT" --vocab "$VOCAB" \
      --device "$DEVICE" --socket "$SOCK" >> "$SRV_LOG" 2>&1 ) &
  srv_pid=$!
  for _ in $(seq 1 90); do          # jusqu'à ~90 s pour charger le modèle
    [ -e "$SOCK.ready" ] && { echo "[watchdog $(date -u +%H:%M:%S)] serveur prêt"; return 0; }
    sleep 1
  done
  echo "[watchdog $(date -u +%H:%M:%S)] ERREUR : serveur non prêt après 90 s" >&2
  return 1
}

serveur_vivant() {
  [ -S "$SOCK" ] && pgrep -f 'infer_server\.py' >/dev/null 2>&1
}

# Garde anti-doublon : deux watchdogs = deux flux = 429.
if [ -f "$PIDFILE" ]; then
  old=$(cat "$PIDFILE" 2>/dev/null)
  if [ -n "$old" ] && kill -0 "$old" 2>/dev/null; then
    echo "[watchdog] un watchdog tourne déjà (PID $old), abandon." >&2
    exit 1
  fi
fi

echo "[watchdog $(date -u +%H:%M:%S)] nettoyage des restes éventuels"
nettoyer
echo $$ > "$PIDFILE"

arreter() {
  echo "[watchdog $(date -u +%H:%M:%S)] arrêt demandé"
  [ -n "$child" ] && kill -- -"$child" 2>/dev/null
  [ -n "$srv_pid" ] && kill "$srv_pid" 2>/dev/null
  sleep 2
  nettoyer
  rm -f "$PIDFILE" "$SOCK" "$SOCK.ready"
  exit 0
}
trap arreter TERM INT

demarrer_serveur || { rm -f "$PIDFILE"; exit 1; }

cd "$BOT_DIR" || exit 1
while true; do
  serveur_vivant || { echo "[watchdog $(date -u +%H:%M:%S)] serveur mort, relance"; demarrer_serveur; }
  echo "[watchdog $(date -u +%H:%M:%S)] démarrage du bot"
  setsid "$PY" lichess-bot.py >> "$LOG" 2>&1 &
  child=$!                      # setsid : chef de session, PGID == $child
  wait "$child"
  code=$?
  kill -- -"$child" 2>/dev/null
  sleep 1
  # ne PAS appeler nettoyer() ici : ça tuerait le serveur ; on ne balaie que le bot
  echo "[watchdog $(date -u +%H:%M:%S)] bot terminé (code $code), redémarrage dans ${RESTART_DELAY}s"
  sleep "$RESTART_DELAY"
done
