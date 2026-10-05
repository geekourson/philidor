"""Étape 3 : le taux de gaffes sur les positions que le bot atteint LUI-MÊME.

Le diagnostic mesurait 18,32 % sur des milieux de partie humains. Mais en jeu,
le modèle visite ses propres positions, pas celles des humains. S'il se perd
dans des positions que les humains n'atteignent pas, son taux de gaffes y sera
nettement plus haut, et la correction devient ciblée : ses propres positions,
annotées par Stockfish, ajoutées à l'affinage.

Source : les PGN écrits partie par partie par lichess-bot. Mêmes critères que
le diagnostic (demi-coups 30 à 80, plus de 12 pièces, partie non terminée),
avec une contrainte de plus : c'est au BOT de jouer. Une seule position par
partie, pour ne pas corréler les échantillons.
"""
import argparse, glob, json, os, random
import chess, chess.pgn

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pgn-dir", default="logs/lichess_games")
    p.add_argument("--bot", default="philidor-142M")
    p.add_argument("--out", default="diag/propres_positions.jsonl")
    p.add_argument("--seed", type=int, default=20260923)
    a = p.parse_args()

    rng = random.Random(a.seed)
    fichiers = sorted(glob.glob(os.path.join(a.pgn_dir, "*.pgn")))
    out, stats = [], {"fichiers": len(fichiers), "variante_ou_fen": 0,
                      "sans_position": 0, "illisible": 0}
    for f in fichiers:
        try:
            g = chess.pgn.read_game(open(f, encoding="utf-8", errors="replace"))
        except Exception:
            stats["illisible"] += 1; continue
        if g is None:
            stats["illisible"] += 1; continue
        h = g.headers
        if h.get("Variant", "Standard") not in ("Standard", "") or "FEN" in h:
            stats["variante_ou_fen"] += 1; continue
        if h.get("White") == a.bot:
            moi, adv = chess.WHITE, "Black"
        elif h.get("Black") == a.bot:
            moi, adv = chess.BLACK, "White"
        else:
            stats["illisible"] += 1; continue
        b = g.board(); coups = []; cands = []
        for i, mv in enumerate(g.mainline_moves()):
            if (30 <= i < 80 and b.turn == moi and not b.is_game_over()
                    and len(b.piece_map()) >= 13):
                cands.append((list(coups), b.fen(), i, mv.uci()))
            coups.append(mv.uci()); b.push(mv)
        if not cands:
            stats["sans_position"] += 1; continue
        pref, fen, ply, joue = rng.choice(cands)
        out.append({"prefixe": pref, "fen": fen, "ply": ply, "coup_joue": joue,
                    "adversaire": h.get(adv, "?"),
                    "adv_bot": h.get(adv + "Title", "") == "BOT",
                    "cadence": h.get("TimeControl", "?"),
                    "classee": h.get("Event", "").lower().startswith("rated"),
                    "partie": os.path.basename(f)})
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as g2:
        for d in out:
            g2.write(json.dumps(d, separators=(",", ":")) + "\n")
    n = len(out)
    print(f"{stats['fichiers']:,} PGN | ecartes : variante/FEN {stats['variante_ou_fen']}, "
          f"sans position eligible {stats['sans_position']}, illisibles {stats['illisible']}")
    print(f"{n:,} positions -> {a.out}")
    print(f"  adversaire BOT : {sum(d['adv_bot'] for d in out):,} | humain : "
          f"{sum(not d['adv_bot'] for d in out):,}")
    print(f"  classees : {sum(d['classee'] for d in out):,}")
    print(f"  demi-coup moyen : {sum(d['ply'] for d in out)/n:.1f}")


if __name__ == "__main__":
    main()
