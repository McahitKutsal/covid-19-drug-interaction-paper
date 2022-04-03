import matplotlib.pyplot as plt
import networkx as nx
from rdkit import Chem

def mol_to_nx(mol):
    G = nx.Graph()

    for atom in mol.GetAtoms():
        G.add_node(atom.GetIdx(),
                   atomic_num=atom.GetAtomicNum(),
                   formal_charge=atom.GetFormalCharge(),
                   chiral_tag=atom.GetChiralTag(),
                   hybridization=atom.GetHybridization(),
                   num_explicit_hs=atom.GetNumExplicitHs(),
                   is_aromatic=atom.GetIsAromatic())
    for bond in mol.GetBonds():
        G.add_edge(bond.GetBeginAtomIdx(),
                   bond.GetEndAtomIdx(),
                   bond_type=bond.GetBondType())
    return G

smiles = 'O=C(O)c1cc(N=Nc2ccc(C=Cc3ccc(N=Nc4cc(C(=O)O)c(O)c5ccccc45)cc3S(=O)(=O)O)c(S(=O)(=O)O)c2)c1O'
import matplotlib.pyplot as plt
import networkx as nx
mol = Chem.MolFromSmiles(smiles)
G = mol_to_nx(mol)
edge_arr = [i for i in G.edges.data()]
edge_attr = {}
for i in edge_arr:
  bond = i[2]['bond_type']
  edge_attr[i[0],i[1]] = bond
edge_list = []

for i in edge_attr:
    edge_list.append(i)

node_labels = {node[0]: node[1]['atomic_num'] for node in G.nodes.data()}
for key, value in node_labels.items():
  if node_labels[key] == 6:
    node_labels[key] = 'C'
  elif node_labels[key] == 8:
    node_labels[key] = 'O'
  elif node_labels[key] == 1:
    node_labels[key] = 'H'
  elif node_labels[key] == 7:
    node_labels[key] = 'N'
  elif node_labels[key] == 16:
    node_labels[key] = 'S'
  elif node_labels[key] == 15:
    node_labels[key] = 'P'
  elif node_labels[key] == 17:
    node_labels[key] = 'Cl'
  elif node_labels[key] == 35:
    node_labels[key] = 'Br'
  elif node_labels[key] == 5:
    node_labels[key] = 'B'
  elif node_labels[key] == 20:
    node_labels[key] = 'Ca'
  elif node_labels[key] == 9:
    node_labels[key] = 'F'
  elif node_labels[key] == 53:
    node_labels[key] = 'I'
  elif node_labels[key] == 11:
    node_labels[key] = 'Na'
edges = edge_list
G = nx.Graph()
G.add_edges_from(edges)
pos = nx.spring_layout(G)
fig = plt.figure(1, figsize=(240, 120),dpi=60)
nx.draw(
    G, pos, edge_color='black', width=2,font_size=220, font_weight='normal', linewidths=1,
    node_size=80000, node_color='pink', alpha=0.9,
    labels=node_labels
)
nx.draw_networkx_edge_labels( 
    G, pos,
    edge_labels=edge_attr,
    font_color='black',
    font_size= 110
)
plt.axis('off')
plt.savefig('visualizations/graph.png')