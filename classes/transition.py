"""
    In this study, the followed tensorflow document was utilized.
    https://keras.io/examples/graph/mpnn-molecular-graphs/
"""


import tensorflow as tf
import numpy as np
from .features import AtomFeaturizer, BondFeaturizer
from rdkit import Chem
np.random.seed(42)
tf.random.set_seed(42)

# We are creating instances of the Classes we are generated in features.py
atom_featurizer = AtomFeaturizer(
    allowable_sets={
        "n_number" : {1, 5, 6, 8, 9, 11, 15, 16, 17, 20, 35, 53}, # Every number corresponds the atomic numbers of B, Br, C, Ca, Cl, F, H, I, N, Na, O, P, S
        "n_valence": {0, 1, 2, 3, 4, 5, 6},
        "n_hydrogens": {0, 1, 2, 3, 4},
        "hybridization": {"s", "sp", "sp2", "sp3"},
        "n_degree": {0, 1, 2, 3, 4, 5, 6},
        "formal_charge": {-2, -1, 0, 1, 2},
        "n_radical_electrons": {0, 1, 2, 3, 4},
        "is_in_ring": {True, False},
        "chiral_tag": {"chi_tetrahedral_cw", "chi_tetrahedral_ccw"},
    }
)

bond_featurizer = BondFeaturizer(
    allowable_sets={
        "bond_type": {"single", "double", "triple", "aromatic"},
        "conjugated": {True, False},
    }
)

def molecule_from_smiles(smiles):
    molecule = Chem.MolFromSmiles(smiles, sanitize=False)
    flag = Chem.SanitizeMol(molecule, catchErrors=True)
    if flag != Chem.SanitizeFlags.SANITIZE_NONE:
        Chem.SanitizeMol(molecule, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ flag)
    Chem.AssignStereochemistry(molecule, cleanIt=True, force=True)
    return molecule

# We are switching from the molecule object to the graph data structure that can be read by the deep learning architecture we will build.
def graph_from_molecule(molecule):
    atom_features = []
    bond_features = []
    pair_indices = []

    for atom in molecule.GetAtoms():
        atom_features.append(atom_featurizer.encode(atom))

        # We make the diagonal in the neighborhood matrix of the graph to be created here completely 1
        # because in the message passing step, its own information will also be created for each node and must be added to the "message"
        pair_indices.append([atom.GetIdx(), atom.GetIdx()])
        bond_features.append(bond_featurizer.encode(None))

        for neighbor in atom.GetNeighbors():
            bond = molecule.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx())
            pair_indices.append([atom.GetIdx(), neighbor.GetIdx()])
            bond_features.append(bond_featurizer.encode(bond))

    return np.array(atom_features), np.array(bond_features), np.array(pair_indices)

# Here we are switching from smiles string to direct graph data structure
def graphs_from_smiles(smiles_list):
    atom_features_list = []
    bond_features_list = []
    pair_indices_list = []

    for smiles in smiles_list:
        molecule = molecule_from_smiles(smiles)
        atom_features, bond_features, pair_indices = graph_from_molecule(molecule)
        
        atom_features_list.append(atom_features)
        bond_features_list.append(bond_features)
        pair_indices_list.append(pair_indices)

    return (
        tf.ragged.constant(atom_features_list, dtype=tf.float32),
        tf.ragged.constant(bond_features_list, dtype=tf.float32),
        tf.ragged.constant(pair_indices_list, dtype=tf.int64),
    )
