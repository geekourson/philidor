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

Trois moteurs, du plus simple au plus fort :
    (défaut)                       la politique seule, une passe, argmax légal
    --valeur juge.pt               la politique propose son top-k, le juge tranche
                                   (juge_recherche/engine_valeur.py)
    --valeur juge.pt --recherche N recherche PUCT, jusqu'à N simulations par coup
                                   (juge_recherche/engine_recherche.py), avec
                                   --graphes pour les graphes CUDA

Avec la recherche, le nombre de simulations est réglé à chaque coup sur la
pendule, le moment de la partie et les parties simultanées
(juge_recherche/budget_pendule.py), jusqu'à une seule passe quand le temps
manque. Le plafond N peut être changé sans redémarrer : il est relu à chaque
coup dans le fichier $PHILIDOR_SIMS_MAX (défaut logs/sims_max.txt). Une
recherche longue s'interrompt si une autre partie attend plus que sa patience.

Protocole, une ligne JSON par sens :
    client -> serveur : {"moves": ["e2e4", "e7e5", ...], "wtime": ..., "btime": ...,
                         "winc": ..., "binc": ...}      (pendule en ms, facultative)
    serveur -> client : {"move": "g1f3", "sims": 1024}   ou   {"error": "..."}
"""

import argparse
import json
import os
import socket
import socketserver
import threading
import time

import chess

from engine import ChessEngine
from juge_recherche.budget_pendule import simulations, palier, patience

# Plafond de simulations modifiable sans redemarrer le bot : lu a chaque coup.
# Absent ou illisible -> la valeur de --recherche.
PLAFOND = os.environ.get("PHILIDOR_SIMS_MAX", os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "sims_max.txt"))


def plafond(defaut):
    try:
        with open(PLAFOND) as f:
            return max(1, int(f.read().strip()))
    except (OSError, ValueError):
        return defaut



class Handler(socketserver.StreamRequestHandler):
    def handle(self):
        for raw in self.rfile:
            try:
                req = json.loads(raw)
                moves = req.get("moves", [])
                board = chess.Board()
                for m in moves:
                    board.push(chess.Move.from_uci(m))
                blanc = board.turn == chess.WHITE
                cle = threading.get_ident()
                with self.server.compteur:
                    self.server.en_cours += 1
                    # heure d'arrivee et attente toleree : consultees par la recherche en cours
                    self.server.attente[cle] = (time.time(), patience(
                        req.get("wtime" if blanc else "btime"), req.get("winc" if blanc else "binc"),
                        len(board.move_stack)))
                try:
                    with self.server.lock:
                        with self.server.compteur:
                            self.server.attente.pop(cle, None)
                        eng = self.server.engine
                        sims = None
                        if self.server.sims_max:
                            # parties simultanees : leurs demandes s'attendent sous le verrou
                            sims = budget(req, board, plafond(self.server.sims_max), self.server.en_cours)
                            eng.sims = sims
                        eng.set_position(board, moves)
                        mv = eng.best_move(board)
                finally:
                    with self.server.compteur:
                        self.server.en_cours -= 1
                        self.server.attente.pop(cle, None)
                resp = {"move": mv.uci()}
                if sims is not None:
                    resp["sims"] = sims
            except Exception as e:  # jamais planter le serveur pour une requête
                resp = {"error": str(e)}
            try:
                self.wfile.write((json.dumps(resp) + "\n").encode())
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                break


def budget(req, board, sims_max, en_cours=1):
    """Simulations de recherche pour ce coup : pendule du camp au trait, moment
    de la partie et parties simultanees (budget_pendule.py). Sans pendule
    (correspondance, analyse) : recherche complete."""
    blanc = board.turn == chess.WHITE
    t = req.get("wtime" if blanc else "btime")
    if t is None:
        mt = req.get("movetime")
        return sims_max if mt is None else palier(mt / max(1, en_cours), sims_max)
    return simulations(t, req.get("winc" if blanc else "binc") or 0, len(board.move_stack),
                       sims_max, en_cours)


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
    # Etape 1 bis : la politique propose son top-k, une copie de run2 affinee en
    # reseau de valeur tranche (engine_valeur.py). Vide = argmax seul.
    p.add_argument("--valeur", default="")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--alpha", type=float, default=0.25)
    # Etape A : recherche PUCT (engine_recherche.py). 0 = pas de recherche.
    p.add_argument("--recherche", type=int, default=0, help="simulations maximum par coup")
    p.add_argument("--cpuct", type=float, default=0.1)
    p.add_argument("--graphes", action="store_true", help="graphes CUDA pour la recherche (graphes.py)")
    args = p.parse_args()

    ready = args.socket + ".ready"
    for path in (args.socket, ready):
        if os.path.exists(path):
            os.unlink(path)

    if args.valeur and args.recherche:
        from juge_recherche.engine_recherche import RechercheEngine
        eng = RechercheEngine(args.ckpt, args.vocab, args.device, args.temperature,
                              sims=args.recherche, cpuct=args.cpuct, fpu=0.0, k=args.k,
                              alpha=args.alpha, valeur_path=args.valeur, graphes=args.graphes)
        if args.graphes:
            # capture de tous les graphes avant d'annoncer le serveur pret
            import time as _t
            t0 = _t.time()
            for L in __import__("juge_recherche.graphes", fromlist=["PALIERS_L"]).PALIERS_L:
                eng.evaluateur([[eng.bos] * L])
            print(f"[moteur] graphes CUDA : {len(eng.evaluateur.graphes)} captures en {_t.time()-t0:.1f} s", flush=True)
        print(f"[moteur] recherche : jusqu'a {args.recherche} simulations selon la pendule, "
              f"cpuct {args.cpuct}, top-{args.k} ; sans recherche : alpha {args.alpha}, "
              f"{args.valeur}", flush=True)
    elif args.valeur:
        from juge_recherche.engine_valeur import ValeurEngine
        eng = ValeurEngine(args.ckpt, args.vocab, args.device, args.temperature,
                           k=args.k, alpha=args.alpha, valeur_path=args.valeur)
        print(f"[moteur] politique + valeur : top-{args.k}, alpha {args.alpha}, "
              f"{args.valeur}", flush=True)
    else:
        eng = ChessEngine(args.ckpt, args.vocab, args.device, args.temperature)
        print("[moteur] standard, sans amorce", flush=True)

    srv = Server(args.socket, Handler)
    srv.engine = eng
    srv.sims_max = args.recherche if (args.valeur and args.recherche) else 0
    srv.compteur = threading.Lock(); srv.en_cours = 0
    srv.attente = {}
    if hasattr(eng, "arret"):
        # une recherche longue s'interrompt si une partie attend depuis plus que
        # sa patience (budget_pendule.patience)
        eng.arret = lambda: any(time.time() - t0 > pat for t0, pat in list(srv.attente.values()))
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
