#!/usr/bin/env python3
import subprocess, textwrap, os
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# [Command Line]
# Run:         python3 run_lambd_sbatch.py
# Show jobs:   squeue -o "%.18i %.9P %.25j %.8u %.2t %.10M %.5D %R"
# Cancel all:  scancel -u "$USER" -p gpu

# -------- user config --------
CURRENT_DATE = date.today().strftime("%b%d")

SEEDS = [456, 789, 123, 101, 112]
LAMBD_VALUES = ["1e-3", "1e-4", "1e-1", "1.0", "5.0", "10.0"]
# -----------------------------

os.makedirs(REPO_ROOT / "logs/slurm", exist_ok=True)

COMMON_SETUP = textwrap.dedent(r"""\
    set -eo pipefail

    module --force purge
    module load math devel
    module load py-wandb/0.18.7_py312
    module load py-pytorch/2.4.1_py312
    module load py-scikit-learn/1.5.1_py312
    module load py-pandas/2.2.1_py312
    source .venv312/bin/activate
""")

LAMBD_WRAP = COMMON_SETUP + textwrap.dedent(r"""\
    python3 run_experiment.py \
        +experiment=lambd_${DATASET} \
        dataset=${DATASET} \
        model=snarenet \
        seed=${SEED} \
        wandb_project=${WANDB_PROJECT} \
        model.lambd=${LAMBD}
""")


def sbatch(job_name, env_vars, wrap_script):
    export = "ALL," + ",".join(f"{k}={v}" for k, v in env_vars.items())
    cmd = [
        "sbatch",
        f"--chdir={REPO_ROOT}",
        "--time=24:00:00",
        "--partition=gpu",
        "-c 1",
        "--gres=gpu:1",
        "--mem=16G",
        f"--job-name={job_name}",
        f"--output=logs/slurm/{job_name}_%j.out",
        f"--error=logs/slurm/{job_name}_%j.err",
        f"--export={export}",
        "--wrap", wrap_script,
    ]
    return subprocess.check_output(cmd, universal_newlines=True).strip().split(";")[0]


jobs = []

dataset = "noncvx"
wandb_project = f"{dataset}_lambd_{CURRENT_DATE}"

for seed in SEEDS:
    for lambd in LAMBD_VALUES:
        lambd_tag = lambd.replace("-", "n").replace(".", "p")
        name = f"snarenet_lambd_{dataset}_s{seed}_l{lambd_tag}"
        env = {
            "DATASET": dataset,
            "SEED": seed,
            "WANDB_PROJECT": wandb_project,
            "LAMBD": lambd,
        }
        jid = sbatch(name, env, LAMBD_WRAP)
        jobs.append(jid)
        print(f"[snarenet-lambd-{dataset}] seed={seed}  lambd={lambd}  -> job {jid}")

print(f"\nSubmitted {len(jobs)} jobs total.")
print(f"wandb_project: {wandb_project}")
