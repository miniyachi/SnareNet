#!/usr/bin/env python3
import subprocess, textwrap, os
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# [Command Line]
# Run:         python3 run_nineq_sbatch.py
# Show jobs:   squeue -o "%.18i %.9P %.25j %.8u %.2t %.10M %.5D %R"
# Cancel all:  scancel -u "$USER" -p gpu

# -------- user config --------
CURRENT_DATE = date.today().strftime("%b%d")

SEEDS = [456, 112, 123, 789, 101]
NUM_INEQ_VALUES = [100, 50, 10]
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

OPT_WRAP = COMMON_SETUP + textwrap.dedent(r"""\
    python3 run_optimizer.py \
        +experiment=opt_${DATASET} \
        dataset=${DATASET} \
        dataset.num_ineq=${NUM_INEQ} \
        wandb_project=${WANDB_PROJECT}
""")

DC3_WRAP = COMMON_SETUP + textwrap.dedent(r"""\
    python3 run_experiment.py \
        +experiment=dc3_ineq${NUM_INEQ} \
        model=dc3 \
        dataset=${DATASET} \
        dataset.num_ineq=${NUM_INEQ} \
        seed=${SEED} \
        wandb_project=${WANDB_PROJECT}
""")

SNARENET_WRAP = COMMON_SETUP + textwrap.dedent(r"""\
    python3 run_experiment.py \
        +experiment=snarenet_ineq \
        dataset=${DATASET} \
        dataset.num_ineq=${NUM_INEQ} \
        model=snarenet \
        seed=${SEED} \
        wandb_project=${WANDB_PROJECT}
""")

HPROJ_WRAP = COMMON_SETUP + textwrap.dedent(r"""\
    python3 run_hproj.py \
        dataset=${DATASET} \
        dataset.num_ineq=${NUM_INEQ} \
        seed=${SEED} \
        wandb_project=${WANDB_PROJECT}
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

dataset = "cvx_qcqp"

for num_ineq in NUM_INEQ_VALUES:
    wandb_project = f"cvx_qcqp_ineq{num_ineq}_{CURRENT_DATE}"

    # Optimizer baseline (no seed)
    name = f"opt_{dataset}_ineq{num_ineq}"
    env = {
        "DATASET": dataset,
        "NUM_INEQ": num_ineq,
        "WANDB_PROJECT": wandb_project,
    }
    jid = sbatch(name, env, OPT_WRAP)
    jobs.append(jid)
    print(f"[opt-{dataset}] num_ineq={num_ineq}  -> job {jid}")

    for seed in SEEDS:
        # DC3
        name = f"dc3_{dataset}_ineq{num_ineq}_s{seed}"
        env = {
            "DATASET": dataset,
            "NUM_INEQ": num_ineq,
            "SEED": seed,
            "WANDB_PROJECT": wandb_project,
        }
        jid = sbatch(name, env, DC3_WRAP)
        jobs.append(jid)
        print(f"[dc3-{dataset}] num_ineq={num_ineq}  seed={seed}  -> job {jid}")

    for seed in SEEDS:
        # SnareNet
        name = f"snarenet_{dataset}_ineq{num_ineq}_s{seed}"
        env = {
            "DATASET": dataset,
            "NUM_INEQ": num_ineq,
            "SEED": seed,
            "WANDB_PROJECT": wandb_project,
        }
        jid = sbatch(name, env, SNARENET_WRAP)
        jobs.append(jid)
        print(f"[snarenet-{dataset}] num_ineq={num_ineq}  seed={seed}  -> job {jid}")

    for seed in SEEDS:
        # HProj
        name = f"hproj_{dataset}_ineq{num_ineq}_s{seed}"
        env = {
            "DATASET": dataset,
            "NUM_INEQ": num_ineq,
            "SEED": seed,
            "WANDB_PROJECT": wandb_project,
        }
        jid = sbatch(name, env, HPROJ_WRAP)
        jobs.append(jid)
        print(f"[hproj-{dataset}] num_ineq={num_ineq}  seed={seed}  -> job {jid}")

print(f"\nSubmitted {len(jobs)} jobs total.")
