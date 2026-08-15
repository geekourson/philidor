"""Serveur d'inférence : LE modèle, chargé une seule fois, pour N parties.

Un modèle sans état n'a aucune raison d'être dupliqué par partie. Ce serveur
charge le modèle une fois et répond aux clients UCI légers (engine_client.py)
via une socket Unix locale. Chaque requête est un historique de coups depuis la
position initiale ; la réponse est le meilleur coup **légal** (le masque de
légalité est appliqué ici, comme dans engine.py).

    python infer_server.py --ckpt checkpoints/run2_best.pt --device cuda:0 \
        --socket logs/infer.sock

Concurrence : chaque connexion est servie dans un thread, mais un verrou
sérialise les passes avant (une à la fois). Ce n'est pas un goulot : une passe
sur ce modèle coûte quelques millisecondes, largement de quoi alimenter des
dizaines de parties en rapid. Le coût mémoire, lui, reste celui d'UN modèle.

Protocole, une ligne JSON par sens :
    client -> serveur : {"moves": ["e2e4", "e7e5", ...]}
    serveur -> client : {"move": "g1f3"}   ou   {"error": "..."}
"""

import argparse
import json
import os
import socket
import socketserver
import threading

import chess

from engine import ChessEngine


class Handler(socketserver.StreamRequestHandler):
    def handle(self):
        for raw in self.rfile:
            try:
                req = json.loads(raw)
                moves = req.get("moves", [])
                board = chess.Board()
                for m in moves:
                    board.push(chess.Move.from_uci(m))
                with self.server.lock:
                    self.server.engine.set_position(board, moves)
                    mv = self.server.engine.best_move(board)
                resp = {"move": mv.uci()}
            except Exception as e:  # jamais planter le serveur pour une requête
                resp = {"error": str(e)}
            try:
                self.wfile.write((json.dumps(resp) + "\n").encode())
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                break


class Server(socketserver.ThreadingUnixStreamServer):
    daemon_threads = True
    allow_reuse_address = True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--vocab", default="data/vocab.json")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--socket", required=True)
    args = p.parse_args()

    ready = args.socket + ".ready"
    for path in (args.socket, ready):
        if os.path.exists(path):
            os.unlink(path)

    eng = ChessEngine(args.ckpt, args.vocab, args.device, args.temperature)

    srv = Server(args.socket, Handler)
    srv.engine = eng
    srv.lock = threading.Lock()   # une passe avant à la fois
    os.chmod(args.socket, 0o600)

    # Signal de disponibilité : le watchdog attend ce fichier avant de lancer
    # lichess-bot, sinon les premiers clients trouvent une socket morte.
    with open(ready, "w") as f:
        f.write("ok\n")
    print(f"infer_server prêt sur {args.socket} (device={args.device})", flush=True)

    try:
        srv.serve_forever()
    finally:
        for path in (args.socket, ready):
            if os.path.exists(path):
                os.unlink(path)


if __name__ == "__main__":
    main()
