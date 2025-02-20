# Download Guacamol dataset from Figshare
import hashlib
import json
import logging
import math
import pathlib
import sys
from typing import List
from urllib.request import urlretrieve

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from scipy.spatial.distance import cdist
from tqdm import tqdm

from da4mt.utils import extract_physicochemical_props


def get_logger():
    logger = logging.getLogger("eamt.perpare.pretraining")
    logger.setLevel(logging.DEBUG)

    print_handler = logging.StreamHandler(stream=sys.stderr)
    print_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s: %(message)s")
    print_handler.setFormatter(formatter)

    logger.addHandler(print_handler)
    return logger


def make_descriptors(output_dir: pathlib.Path):
    logger = get_logger()
    train_smiles_file = output_dir / "guacamol_v1_train.smiles"
    validation_smiles_file = output_dir / "guacamol_v1_valid.smiles"

    train_smiles = train_smiles_file.read_text().splitlines(keepends=False)
    validation_smiles = validation_smiles_file.read_text().splitlines(keepends=False)

    # Path to normalization values (mean, std) for physicochemical properties
    normalization_out = output_dir / "guacamol_v1_normalization_values.json"
    # Path to the labeled dataset
    train_out = output_dir / "guacamol_v1_train_mtr.jsonl"
    validation_out = output_dir / "guacamol_v1_valid_mtr.jsonl"

    logger.info("Extracting physicochemical properties from the Train dataset.")
    extract_physicochemical_props(
        train_smiles, train_out, logger, normalization_path=normalization_out
    )

    logger.info("Extracting physicochemical properties from the Validation dataset.")
    extract_physicochemical_props(
        validation_smiles, validation_out, logger, normalization_path=None
    )


def download_guacamole(output_dir: pathlib.Path):
    logger = get_logger()

    # Guacamol dataset download Urls, taken from to official guacamol repo
    # https://github.com/BenevolentAI/guacamol/blob/60ebe1f6a396f16e08b834dce448e9343d259feb/README.md#download
    dataset_name = "guacamol_v1"
    urls = [
        "https://ndownloader.figshare.com/files/13612760",
        "https://ndownloader.figshare.com/files/13612766",
        "https://ndownloader.figshare.com/files/13612757",
    ]
    # md5 checksums to verify the downloads
    checksums = [
        "05ad85d871958a05c02ab51a4fde8530",
        "e53db4bff7dc4784123ae6df72e3b1f0",
        "677b757ccec4809febd83850b43e1616",
    ]
    file_names = ["train.smiles", "valid.smiles", "test.smiles"]

    for url, checksum, split_name in tqdm(zip(urls, checksums, file_names)):
        logger.info(f"Downloading {split_name} from {url}")
        output_file = output_dir / f"{dataset_name}_{split_name}"
        urlretrieve(url, output_file)

        with open(output_file, "rb") as f:
            if hashlib.md5(f.read()).hexdigest() != checksum:
                raise ValueError(
                    "Checksums don't match. There likely was an error during downloading."
                )


def select_diverse_sample(
    fingerprints: np.ndarray,
    centroids: List[List[int]],
    clusters: List[List[int]],
    size: int,
) -> List[int]:
    """Select a diverse sample of molecules based on cluster centroids.

    Samples molecules from each cluster, prioritizing larger clusters and
    molecules closest to cluster centroids. The number of samples from each
    cluster is either 10 or 50% of cluster size, whichever is smaller.

    :param fingerprints: Array of Morgan fingerprints for all molecules
    :param centroids: Cluster centroids ordered by descending cluster size
    :param clusters: Lists of indices for each cluster, matching centroid order
    :param size: Number of molecules to select
    :raises ValueError: If requested sample size exceeds available molecules
    :return: List of selected molecule indices
    """
    if size > len(fingerprints):
        raise ValueError(
            f"Requested sample size ({size}) exceeds available molecules ({len(fingerprints)})"
        )

    result = []
    remaining = size - len(result)
    index = 0
    while remaining > 0:
        assert index < len(clusters)
        cluster = clusters[index]
        if len(cluster) > 10:
            n_samples = 10
        else:
            n_samples = int(0.5 * len(cluster)) + 1

        n_samples = min(remaining, n_samples)

        # We take the most similar samples to the cluster centroid
        centroid = centroids[index]
        similarities = (
            1 - cdist(fingerprints[cluster], [centroid], metric="jaccard").flatten()
        )
        # Most similar first
        sim_with_idx = sorted(
            [(sim, idx) for sim, idx in zip(similarities, cluster)], reverse=True
        )
        result.extend([idx for _, idx in sim_with_idx[:n_samples]])

        index += 1
        remaining -= n_samples

    return result


def make_clusters(output_dir: pathlib.Path):
    from external.bitbirch import BitBirch

    logger = get_logger()
    with open(output_dir / "guacamol_v1_train.smiles", "r") as f:
        smiles = f.read().splitlines(keepends=False)

    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

    fps = []
    invalid = set()
    for line, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol:
            fps.append(mfpgen.GetFingerprintAsNumPy(mol))
        else:
            invalid.add(line)
            logger.warning(f"Molecule '{smi}' at line {line} is not valid.")

    logger.info(f"{len(invalid)} invalid molecules found.")

    np_fps = np.stack(fps).astype(np.int64)
    bb = BitBirch()
    bb.fit(np_fps)

    # We sort the clusters and centroids by size
    clusters: List[List[int]] = bb.get_cluster_mol_ids()
    centroids: List[np.ndarray] = bb.get_centroids()
    sizes = [len(c) for c in clusters]

    # From largest to smallest
    sorted_indices = np.argsort(sizes)[::-1]
    clusters = [clusters[i] for i in sorted_indices]
    centroids: List[List[int]] = [centroids[i].tolist() for i in sorted_indices]

    outfile = output_dir / "guacamol_train_clusters.json"
    logger.info(f"Saving clusters to '{outfile}'")
    with open(outfile, "w") as f:
        output = {"clusters": clusters, "centroids": centroids}
        json.dump(output, f)

    # Print some statistics
    logger.debug(f"Data has {len(clusters)} clusters.")
    logger.debug(f"The largest cluster has {max(sizes)} samples.")
    logger.debug(f"{sum([1 for s in sizes if s == 1])} singleton clusters.")

    # Select fractions for pretraining by sampling from
    # possibly all clusters, but giving priority to the larger
    # clusters
    for fraction in [0.3, 0.6]:
        size = math.ceil(len(smiles) * fraction)
        logger.info(f"Selecting a diverse sample of {size} molecules.")
        selected = select_diverse_sample(np_fps, centroids, clusters, size)

        subset_file = output_dir / f"guacamol_train_clusters_{int(fraction*100)}.json"
        logger.info(f"Saving clusters sample to '{subset_file}'")
        with open(subset_file, "w") as f:
            json.dump(selected, f)


def make_pretraining(args):
    args.output_dir.mkdir(exist_ok=True)

    download_guacamole(args.output_dir)
    make_descriptors(args.output_dir)
    make_clusters(args.output_dir)
