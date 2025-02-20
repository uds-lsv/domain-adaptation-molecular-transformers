import os
import pathlib
import subprocess
import sys
from typing import List, NamedTuple


class Paths(NamedTuple):
    DATA_DIR: pathlib.Path
    RESULT_DIR: pathlib.Path
    MODEL_DIR: pathlib.Path
    PRETRAIN_DIR: pathlib.Path
    ADAPT_DIR: pathlib.Path
    WANDB_DIR: pathlib.Path
    EMBED_DIR: pathlib.Path


PATHS = Paths(
    DATA_DIR=pathlib.Path("/data/users/mrdupont/da4mt/data/"),
    # RESULT_DIR=pathlib.Path("/nethome/mrdupont/enumeration-aware-molecule-transformers/results"),
    # RESULT_DIR=pathlib.Path("/nethome/mrdupont/enumeration-aware-molecule-transformers/results/results_with_clustered_pretraining_cleaned_adme_microsom"),
    RESULT_DIR=pathlib.Path("/nethome/mrdupont/enumeration-aware-molecule-transformers/results/results_with_clustered_pretraining_cleaned_adme_microsom_including_mtr_mtr_30"),
    MODEL_DIR=pathlib.Path("/data/users/mrdupont/da4mt/models"),
    # PRETRAIN_DIR=pathlib.Path("/data/users/mrdupont/da4mt/models/pretrained"),
    PRETRAIN_DIR=pathlib.Path("/data/users/mrdupont/da4mt/models/pretrained_cluster"),
    # ADAPT_DIR=pathlib.Path("/data/users/mrdupont/da4mt/models/adapted"),
    ADAPT_DIR=pathlib.Path("/data/users/mrdupont/da4mt/models/adapted_cluster"),
    WANDB_DIR=pathlib.Path("/data/users/mrdupont/da4mt/wandb"),
    EMBED_DIR=pathlib.Path("/data/users/mrdupont/da4mt/embeddings")
)

ADME_DATASETS = [
    "adme_microsom_stab_h",
    "adme_microsom_stab_r",
    "adme_permeability",
    "adme_ppb_h",
    "adme_ppb_r",
    "adme_solubility",
]

CHEMBENCH_DATASETS = [
    "bace",
    "bbbp",
    "clintox",
    "sider",
    "toxcast",
    "esol",
    "lipop",
    "freesolv",
    "hiv",
]

DATASETS = ADME_DATASETS + CHEMBENCH_DATASETS


def rename_gpus() -> None:
    """
    Rename GPUs to access them via CUDA_VISIBLE_DEVICES.

    This function modifies the CUDA_VISIBLE_DEVICES environment variable
    to use the actual GPU indices as reported by nvidia-smi.
    """
    cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not cuda_devices:
        return

    new_devices = []
    for device_id in cuda_devices.split(","):
        cmd = f"nvidia-smi -L | grep {device_id} | sed -E 's/^GPU ([0-9]+):.*$/\\1/'"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.stdout.strip():
            new_devices.append(result.stdout.strip())

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(new_devices)


def setup_environment(kwargs) -> None:
    """
    Set up the environment variables for various directory paths.
    """
    rename_gpus()

    for key, value in kwargs.items():
        os.environ[key] = value


def load_wandb_credentials() -> None:
    """
    Load Weights & Biases login credentials from .netrc file.
    """
    cmd = "awk '/machine api\.wandb\.ai/,/^\*$/ { if (/password/) print $2}' /nethome/mrdupont/.netrc"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    api_key = result.stdout.strip()
    os.environ["WANDB_API_KEY"] = api_key
    print(f"Found WANDB_API_KEY={api_key}")


def print_debug_info() -> None:
    """
    Print debug information about the current environment.
    """
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
    print("\nEnvironment variables:")
    for key, value in os.environ.items():
        print(f"\t{key}={value}")

    print(f"Current directory: {os.getcwd()}")
    print(f"Arguments: {sys.argv}")


def change_to_project_root() -> None:
    """
    Change the current working directory to PROJECT_ROOT.
    """
    project_root = os.environ.get("PROJECT_ROOT")
    if project_root:
        os.chdir(project_root)
    else:
        print("WARNING: PROJECT_ROOT environment variable is not set.")


def run_command(cmd: List[str]) -> None:
    """
    Run a command and handle potential errors.

    :param cmd: Command to run as a list of strings
    """
    try:
        print(f"Running command: {' '.join(cmd)}", file=sys.stderr)
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error: Command failed with return code {e.returncode}", file=sys.stderr)
        sys.exit(1)


# Execute setup functions globally
setup_environment({k: str(v) for k,v in PATHS._asdict().items()})
load_wandb_credentials()
change_to_project_root()
print_debug_info()

print("Leaving setup_cluster.py")
