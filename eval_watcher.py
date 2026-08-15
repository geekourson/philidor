"""Phase 3 : Évaluer les instantanés au fil de l'entraînement, sur l'autre carte.

Pendant que le 3090 entraîne, ce script surveille l'apparition d'instantanés
numérotés et les évalue sur le 3060. On obtient ainsi, sans ralentir
l'entraînement d'une seconde, la courbe qui montre le modèle découvrir les
règles du jeu : le taux de coups légaux en fonction du nombre de tokens vus.

C'est le graphique le plus parlant du projet, et il a une particularité, on ne
peut pas le reconstituer après coup. Si on n'évalue pas les états
intermédiaires pendant qu'ils existent, ils sont perdus. D'où ce script, lancé
en parallèle du run.

Usage :
    nohup python eval_watcher.py --run-name run1 > logs/eval_watcher.log 2>&1 &
"""

import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time

STEP_RE = re.compile(r"_step(\d+)\.pt$")


def find_snapshots(ckpt_dir, run_name):
    pattern = os.path.join(ckpt_dir, f"{run_name}_step*.pt")
    out = []
    for path in glob.glob(pattern):
        m = STEP_RE.search(path)
        if m:
            out.append((int(m.group(1)), path))
    return sorted(out)


def already_done(log_dir, step, prefix="eval"):
    return os.path.exists(os.path.join(log_dir, f"{prefix}_step{step}.json"))


def evaluate_one(python, step, path, args):
    """Lance evaluate.py en sous-processus.

    Pourquoi un sous-processus plutôt qu'un import : chaque évaluation charge
    un modèle sur le GPU, et PyTorch ne rend jamais complètement la mémoire
    d'un modèle libéré au sein d'un même processus. Sur vingt évaluations
    successives, la fragmentation finirait par provoquer un dépassement
    mémoire. Un processus qui meurt rend tout, sans exception.
    """
    cmd = [
        python, "evaluate.py",
        "--ckpt", path,
        "--device", args.device,
        "--vocab", args.vocab,
        "--val-games", args.val_games,
        "--n-legal", str(args.n_legal),
        "--n-agreement", str(args.n_agreement),
        "--n-per-rule", str(args.n_per_rule),
        "--n-full-games", str(args.n_full_games),
        "--out", os.path.join(args.log_dir, f"{args.prefix}_step{step}.json"),
        "--temperature", str(args.temperature),
    ]
    print(f"[{time.strftime('%H:%M:%S')}] évaluation du step {step}...",
          flush=True)
    t0 = time.perf_counter()
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  ECHEC (code {r.returncode})", flush=True)
        print(r.stdout[-2000:], flush=True)
        print(r.stderr[-2000:], flush=True)
        return False
    # On ne réaffiche que la ligne du chiffre central.
    for line in r.stdout.splitlines():
        if "%" in line and "[IC95" in line:
            print(f"  {line.strip()}", flush=True)
            break
    print(f"  terminé en {time.perf_counter()-t0:.0f} s", flush=True)
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-name", default="run1")
    p.add_argument("--ckpt-dir", default="checkpoints")
    p.add_argument("--log-dir", default="logs")
    p.add_argument("--device", default="cuda:0", help="cuda:0 = RTX 3060")
    p.add_argument("--vocab", default="data/vocab.json")
    p.add_argument("--val-games", default="data/val_games.txt")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--poll-seconds", type=int, default=120)
    # Sans préfixe distinct, les rapports de deux runs différents s'écrasent
    # mutuellement dans logs/. C'est arrivé de justesse entre run1 et run2.
    p.add_argument("--prefix", default="eval",
                   help="préfixe des rapports, ex. eval_run2")
    p.add_argument("--temperature", type=float, default=1.0,
                   help="température de génération pour les parties complètes")
    # Échantillons volontairement réduits par rapport à l'évaluation finale :
    # ici on veut une courbe, pas un chiffre publiable. L'évaluation finale
    # du meilleur checkpoint utilisera les tailles par défaut, bien plus
    # grandes, pour resserrer les intervalles de confiance.
    p.add_argument("--n-legal", type=int, default=2000)
    p.add_argument("--n-agreement", type=int, default=2000)
    p.add_argument("--n-per-rule", type=int, default=150)
    p.add_argument("--n-full-games", type=int, default=60)
    # 150 min et non 45 : ce délai doit dépasser confortablement l'intervalle
    # entre deux instantanés, sinon le watcher expire entre deux et s'arrête
    # en plein run. Sur run2, un instantané tous les 5000 steps à 50 000
    # tokens/s fait 68 minutes, le réglage initial de 45 min garantissait
    # l'arrêt prématuré. Vérifier ce rapport à chaque changement de cadence
    # d'instantanés ou de débit.
    p.add_argument("--stop-when-idle-minutes", type=float, default=150.0,
                   help="s'arrête si aucun nouvel instantané pendant ce délai ; "
                        "doit dépasser l'intervalle entre deux instantanés")
    args = p.parse_args()

    print(f"Surveillance de {args.ckpt_dir}/{args.run_name}_step*.pt "
          f"sur {args.device}", flush=True)
    last_activity = time.perf_counter()

    while True:
        snapshots = find_snapshots(args.ckpt_dir, args.run_name)
        pending = [(s, p_) for s, p_ in snapshots
                   if not already_done(args.log_dir, s, args.prefix)]

        if pending:
            for step, path in pending:
                # On attend que le fichier soit stable : l'écriture est
                # atomique (rename), mais mieux vaut ne pas courir après.
                size = os.path.getsize(path)
                time.sleep(2)
                if os.path.getsize(path) != size:
                    continue
                evaluate_one(args.python, step, path, args)
                last_activity = time.perf_counter()
        else:
            idle_min = (time.perf_counter() - last_activity) / 60
            if idle_min >= args.stop_when_idle_minutes:
                print(f"Aucun nouvel instantané depuis {idle_min:.0f} min, "
                      f"arrêt de la surveillance.", flush=True)
                break
            time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
