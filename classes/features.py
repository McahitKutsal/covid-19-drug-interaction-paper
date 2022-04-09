"""
    In this study, the followed tensorflow document was utilized.
    https://keras.io/examples/graph/mpnn-molecular-graphs/
"""

import numpy as np

class Featurizer:
    def __init__(self, allowable_sets):
        self.dim = 0
        self.features_mapping = {}
        for k, s in allowable_sets.items():
            s = sorted(list(s))
            self.features_mapping[k] = dict(zip(s, range(self.dim, len(s) + self.dim)))
            self.dim += len(s)

    def encode(self, inputs):
        output = np.zeros((self.dim,))
        for name_feature, feature_mapping in self.features_mapping.items():
            feature = getattr(self, name_feature)(inputs)
            if feature not in feature_mapping:
                continue
            output[feature_mapping[feature]] = 1.0
        return output


class AtomFeaturizer(Featurizer):
    def __init__(self, allowable_sets):
        super().__init__(allowable_sets)

    # The number of valence electrons is the number of chemical bonds that atoms of an element can form.
    # In other words, it is a measure of the ability of an element to combine.
    def n_valence(self, atom):
        return atom.GetTotalValence()

    # The atomic number of chemical elements is the number of protons in the nucleus of each atom of that element.
    def n_number(self, atom):
        return atom.GetAtomicNum()

    # The degree of an atom is defined to be its number of directly-bonded neighbors. 
    # The degree is independent of bond orders.
    def n_degree(self, atom):
        return atom.GetDegree()

    # In chemistry, a formal charge in the covalent view of bonding,
    # is the charge assigned to an atom in a molecule, assuming that electrons in all chemical bonds are shared equally between atoms, 
    # regardless of relative electronegativity
    def formal_charge(self, atom):
        return atom.GetFormalCharge()

    # They are often represented as resonance structures containing single and double bonds.
    def is_aromatic(self, atom):
        return atom.GetIsAromatic()
    
    # Radicals are the species which contain at least one unpaired electron in the shells
    # around the atomic nucleus and are capable of independent existence.
    def n_radical_electrons(self, atom):
        return atom.GetNumRadicalElectrons()
    
    # In chemistry, a ring is an ambiguous term referring either to a simple cycle of atoms and bonds in a molecule
    #  or to a connected set of atoms and bonds in which every atom and bond is a member of a cycle (also called a ring system).
    def is_in_ring(self, atom):
        return atom.IsInRing()

    # Chirality essentially means mirror-image, non-superimposable molecules
    def chiral_tag(self, atom):
        return atom.GetChiralTag().name.lower()
    
    def n_hydrogens(self, atom):
        return atom.GetTotalNumHs()

    # Hybridization is a chemical process where different orbitals of an atom form the same hybrid orbitals.
    def hybridization(self, atom):
        return atom.GetHybridization().name.lower()


class BondFeaturizer(Featurizer):
    def __init__(self, allowable_sets):
        super().__init__(allowable_sets)
        self.dim += 1

    def encode(self, bond):
        output = np.zeros((self.dim,))
        if bond is None:
            output[-1] = 1.0
            return output
        output = super().encode(bond)
        return output

    # A chemical bond is the force that binds atoms together and keeps them together.
    def bond_type(self, bond):
        return bond.GetBondType().name.lower()

    # In chemistry, a conjugated system is a system of p-orbitals that lowers the overall energy of the molecule and increases stability.
    def conjugated(self, bond):
        return bond.GetIsConjugated()
