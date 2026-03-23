#!/usr/bin/env python3
import subprocess, textwrap, os
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# [Command Line]
# Run:         python3 run_noncvx_sbatch.py
# Show jobs:   squeue -o "%.18i %.9P %.25j %.8u %.2t %.10M %.5D %R"
# Cancel all:  scancel -u "$USER" -p gpu

# -------- user config --------
CURRENT_DATE = date.today().strftime("%b%d")

SEEDS = [456, 789, 123, 101, 112]
RTOL_VALUES = ["1e-4", "1e-6", "1e-8", "1e-10", "1e-12"]
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

NONCVX_WRAP = COMMON_SETUP + textwrap.dedent(r"""\
    python3 run_experiment.py \
        +experiment=${MODEL}_${DATASET} \
        dataset=${DATASET} \
        model=${MODEL} \
        seed=${SEED} \
        wandb_project=${WANDB_PROJECT} \
        ${EXTRA_ARGS}
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
wandb_project = f"{dataset}_comparison_{CURRENT_DATE}"

for seed in SEEDS:
    # HardNetAff
    name = f"hardnetaff_{dataset}_s{seed}"
    env = {
        "MODEL": "hardnetaff",
        "DATASET": dataset,
        "SEED": seed,
        "WANDB_PROJECT": wandb_project,
        "EXTRA_ARGS": "",
    }
    jid = sbatch(name, env, NONCVX_WRAP)
    jobs.append(jid)
    print(f"[hardnetaff-{dataset}] seed={seed}  -> job {jid}")

    # DC3
    name = f"dc3_{dataset}_s{seed}"
    env = {
        "MODEL": "dc3",
        "DATASET": dataset,
        "SEED": seed,
        "WANDB_PROJECT": wandb_project,
        "EXTRA_ARGS": "",
    }
    jid = sbatch(name, env, NONCVX_WRAP)
    jobs.append(jid)
    print(f"[dc3-{dataset}] seed={seed}  -> job {jid}")

    # SnareNet with different rtol values
    for rtol in RTOL_VALUES:
        rtol_tag = rtol.replace("-", "n")
        name = f"snarenet_{dataset}_s{seed}_rtol{rtol_tag}"
        env = {
            "MODEL": "snarenet",
            "DATASET": dataset,
            "SEED": seed,
            "WANDB_PROJECT": wandb_project,
            "EXTRA_ARGS": f"model.rtol={rtol}",
        }
        jid = sbatch(name, env, NONCVX_WRAP)
        jobs.append(jid)
        print(f"[snarenet-{dataset}] seed={seed}  rtol={rtol}  -> job {jid}")

print(f"\nSubmitted {len(jobs)} jobs total.")
print(f"wandb_project: {wandb_project}")
