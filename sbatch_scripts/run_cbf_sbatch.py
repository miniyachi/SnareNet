#!/usr/bin/env python3
import subprocess, textwrap, os
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# [Command Line]
# Run:         python3 run_cbf_sbatch.py
# Show jobs:   squeue -o "%.18i %.9P %.25j %.8u %.2t %.10M %.5D %R"
# Cancel all:  scancel -u "$USER" -p gpu

# -------- user config --------
CURRENT_DATE = date.today().strftime("%b%d")

EXPERIMENTS = [
    {"dataset_seed": 8, "seed": 123},
    {"dataset_seed": 9, "seed": 123},
]
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

CBF_WRAP = COMMON_SETUP + textwrap.dedent(r"""\
    python3 run_experiment.py \
        +experiment=${MODEL}_cbf \
        dataset=cbf \
        model=${MODEL} \
        dataset.seed=${DATASET_SEED} \
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

wandb_project = f"cbf_base_comparison_{CURRENT_DATE}"

for model in ["dc3", "snarenet"]:
    for exp in EXPERIMENTS:
        dataset_seed = exp["dataset_seed"]
        seed = exp["seed"]
        name = f"{model}_cbf_ds{dataset_seed}_s{seed}"
        env = {
            "MODEL": model,
            "DATASET_SEED": dataset_seed,
            "SEED": seed,
            "WANDB_PROJECT": wandb_project,
        }
        jid = sbatch(name, env, CBF_WRAP)
        jobs.append(jid)
        print(f"[{model}-CBF] dataset_seed={dataset_seed}  seed={seed}  -> job {jid}")

print(f"\nSubmitted {len(jobs)} jobs total.")
print(f"wandb_project: {wandb_project}")
