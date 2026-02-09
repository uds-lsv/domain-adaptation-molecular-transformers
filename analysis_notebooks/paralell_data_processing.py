from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd

from rdkit import Chem

from rdkit.Chem import Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

descriptors_list = ['MW', 'TPSA', 'LogP', 'HBA', 'HBD', 'RotatableBonds']


def calculate_descriptors(molecule):
    descriptors = {}
    descriptors['MW'] = Descriptors.MolWt(molecule)
    descriptors['TPSA'] = Descriptors.TPSA(molecule)
    descriptors['LogP'] = Descriptors.MolLogP(molecule)
    descriptors['HBA'] = Descriptors.NumHAcceptors(molecule)
    descriptors['HBD'] = Descriptors.NumHDonors(molecule)
    descriptors['RotatableBonds'] = Descriptors.NumRotatableBonds(molecule)
    return descriptors


def extract_scaffold(mol):
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    scaffold_smiles = Chem.MolToSmiles(scaffold)
    return scaffold_smiles


def process_molecule(row):
    """Process a single SMILES string to compute descriptors and scaffold."""
    smiles = row['SMILES']
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    row['molecules'] = mol
    row['Descriptors'] = calculate_descriptors(mol)
    for desc in descriptors_list:
        row[desc] = row['Descriptors'][desc]
    row['Scaffold'] = extract_scaffold(mol)
    return row


def parallel_apply(data, func, n_cores=None):
    """Apply a function to a Pandas DataFrame in parallel."""
    n_cores = n_cores or cpu_count()
    with Pool(n_cores) as pool:
        chunks = np.array_split(data, n_cores)
        results = pool.map(func, chunks)
    return pd.concat(results)


def process_molecules_in_parallel(data):
    """Process molecules to add descriptors and scaffolds."""
    return data.apply(process_molecule, axis=1)



