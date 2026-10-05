"""Moteur UCI ultra-léger : NE charge pas le modèle, relaie vers infer_server.

lichess-bot lance un moteur par partie. Pour un modèle sans état, charger une
copie du modèle par partie est un gaspillage : ce client ne fait que parler le
protocole UCI et transmettre l'historique de coups au serveur d'inférence
(infer_server.py), qui détient l'unique copie du modèle.

Conséquences :
  - démarrage **instantané** (aucun import de torch, aucun chargement de poids) ;
  - empreinte minime (~quelques dizaines de Mo) ;
  - la concurrence n'est plus limitée par la mémoire mais par lichess-bot et
    Lichess. 50 clients légers + 1 modèle, au lieu de 50 modèles.

Le serveur applique le masque de légalité, donc ce client ne renvoie jamais un
coup illégal. Socket configurable via la variable d'environnement INFER_SOCKET.

Limitation identique à engine.py : seul `position startpos moves ...` est
géré (le modèle a besoin de l'historique, pas d'une FEN).
"""

import json
import os
import socket
import sys
import time

SOCK = os.environ.get("INFER_SOCKET",
                      "logs/infer.sock")


def demander(moves, pendule=None, tentatives=40):
    """Envoie l'historique au serveur, renvoie le coup. Réessaie si le serveur
    redémarre (le watchdog peut le relancer sous nous)."""
    dernier = None
    for _ in range(tentatives):
        try:
            s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            s.settimeout(30)
            s.connect(SOCK)
            s.sendall((json.dumps({"moves": moves, **(pendule or {})}) + "\n").encode())
            data = b""
            while not data.endswith(b"\n"):
                chunk = s.recv(4096)
                if not chunk:
                    break
                data += chunk
            s.close()
            resp = json.loads(data)
            if resp.get("move"):
                return resp["move"]
            dernier = resp.get("error", "réponse vide")
        except (FileNotFoundError, ConnectionRefusedError, OSError) as e:
            dernier = str(e)
            time.sleep(0.5)   # serveur peut-être en train de (re)démarrer
    raise RuntimeError(f"serveur d'inférence injoignable : {dernier}")


def main():
    moves = []
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        cmd = parts[0]

        if cmd == "uci":
            print("id name philidor-142M")
            print("id author Billy Girboux")
            print("uciok", flush=True)
        elif cmd == "isready":
            print("readyok", flush=True)
        elif cmd == "ucinewgame":
            moves = []
        elif cmd == "position":
            if "startpos" in parts and "moves" in parts:
                moves = parts[parts.index("moves") + 1:]
            elif "startpos" in parts:
                moves = []
            # `position fen ...` non supporté : le modèle a besoin de l'historique
        elif cmd == "go":
            # la pendule (ms) sert au serveur à régler l'effort de recherche
            pendule = {}
            for cle in ("wtime", "btime", "winc", "binc", "movetime"):
                if cle in parts:
                    try:
                        pendule[cle] = int(parts[parts.index(cle) + 1])
                    except (ValueError, IndexError):
                        pass
            print(f"bestmove {demander(moves, pendule)}", flush=True)
        elif cmd in ("quit", "stop"):
            if cmd == "quit":
                break


if __name__ == "__main__":
    main()
