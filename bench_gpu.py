"""Phase 0 : Vérification de l'environnement et benchmark matmul bf16.

Mesure les TFLOPS bf16 réels de chaque GPU visible, afin de pouvoir estimer
plus tard la durée de l'entraînement (via le MFU : Model FLOPs Utilization).

On force CUDA_DEVICE_ORDER=PCI_BUS_ID pour que les index PyTorch correspondent
exactement à ceux affichés par nvidia-smi (sinon PyTorch trie par "GPU le plus
rapide d'abord", ce qui prête à confusion quand on a deux cartes différentes).
"""

import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import torch


def nvidia_smi_table():
    out = subprocess.run(
        ["nvidia-smi",
         "--query-gpu=index,name,memory.total,memory.used,memory.free",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True).stdout.strip()
    rows = []
    for line in out.splitlines():
        idx, name, total, used, free = [p.strip() for p in line.split(",")]
        rows.append({"index": int(idx), "name": name,
                     "memory_total_mib": int(total),
                     "memory_used_mib": int(used),
                     "memory_free_mib": int(free)})
    return rows


def bench_matmul(device_idx, n=8192, dtype=torch.bfloat16, warmup=5, iters=30):
    """Multiplie deux matrices n x n et renvoie le débit en TFLOPS.

    Un produit matriciel n x n coûte 2*n^3 opérations flottantes (n^3
    multiplications + n^3 additions). En mesurant le temps, on en déduit le
    nombre d'opérations par seconde que la carte soutient réellement --
    toujours bien en dessous du chiffre marketing du constructeur.
    """
    dev = torch.device(f"cuda:{device_idx}")
    free_bytes = torch.cuda.mem_get_info(dev)[0]
    # 3 matrices (a, b, résultat) de n*n éléments sur 2 octets, + marge x2
    need = 3 * n * n * 2 * 2
    while need > free_bytes and n > 1024:
        n //= 2
        need = 3 * n * n * 2 * 2

    a = torch.randn(n, n, device=dev, dtype=dtype)
    b = torch.randn(n, n, device=dev, dtype=dtype)

    for _ in range(warmup):
        c = a @ b
    torch.cuda.synchronize(dev)

    t0 = time.perf_counter()
    for _ in range(iters):
        c = a @ b
    torch.cuda.synchronize(dev)
    dt = time.perf_counter() - t0

    flops = 2 * (n ** 3) * iters
    tflops = flops / dt / 1e12
    del a, b, c
    torch.cuda.empty_cache()
    return {"matrix_size": n, "iters": iters, "seconds": round(dt, 4),
            "tflops_bf16": round(tflops, 2)}


def main():
    report = {
        "date_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
        "cudnn": torch.backends.cudnn.version(),
        "cuda_device_order": os.environ.get("CUDA_DEVICE_ORDER"),
        "gpus": [],
        "nvidia_smi": nvidia_smi_table(),
    }

    if not report["cuda_available"]:
        print(json.dumps(report, indent=2))
        sys.exit("ECHEC : CUDA indisponible.")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        free, total = torch.cuda.mem_get_info(i)
        entry = {
            "torch_index": i,
            "name": props.name,
            "capability": f"{props.major}.{props.minor}",
            "sm_count": props.multi_processor_count,
            "vram_total_mib": total // 2**20,
            "vram_free_mib": free // 2**20,
        }
        print(f"[bench] cuda:{i} {props.name} "
              f"({entry['vram_free_mib']} MiB libres) ...", flush=True)
        entry["matmul_bf16"] = bench_matmul(i)
        print(f"        -> {entry['matmul_bf16']['tflops_bf16']} TFLOPS bf16 "
              f"(matrices {entry['matmul_bf16']['matrix_size']}^2)", flush=True)
        report["gpus"].append(entry)

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "logs", "phase0_bench.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nRapport ecrit dans {out}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
