import os
import torch
import openbabel
from openbabel import pybel
import warnings
warnings.filterwarnings('ignore')
from torch_geometric.data import Data,HeteroData
from scipy.spatial import distance_matrix
import torch_geometric.transforms as T
import pickle
from tqdm import tqdm
from transformers import pipeline
import re
from prody import *
import networkx as nx
import numpy as np
from transformers import BertModel, BertTokenizer
from transformers import T5Tokenizer, T5EncoderModel
import os
from Bio.PDB import *
from Bio import PDB
import atom3d.util.formats as fo
from utils.protein_utils import featurize_as_graph, featurize_as_graph_dssp
from utils.openbabel_featurizer import Featurizer
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen
from Bio.Seq import Seq
from Bio.PDB import PDBParser
from Bio.SeqUtils import ProtParam
from scipy.spatial.distance import pdist, squareform
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
import torch_cluster

seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

ele2num = {'H':0, 'LI':1, 'C':2, 'N':3, 'O':4, 'NA':5, 'MG':6, 'P':7, 'S':8, 'K':9, 'CA':10, 'MN':11,
           'FE':12, 'CO':13, 'NI':14, 'CU':15, 'ZN':16, 'SE':17, 'SR':18, 'CD':19, 'CS':20, 'HG':21}

device = torch.device('cuda')

def info_3D(a, b, c):
    ab = b - a
    ac = c - a
    cosine_angle = np.dot(ab, ac) / (np.linalg.norm(ab) * np.linalg.norm(ac))
    cosine_angle = cosine_angle if cosine_angle >= -1.0 else -1.0
    angle = np.arccos(cosine_angle)
    ab_ = np.sqrt(np.sum(ab ** 2))
    ac_ = np.sqrt(np.sum(ac ** 2))
    area = 0.5 * ab_ * ac_ * np.sin(angle)
    return np.degrees(angle), area, ac_

def info_3D_cal(edge, ligand,h_num):
    node1_idx = edge[0]
    node2_idx = edge[1]
    atom1 = ligand.atoms[node1_idx]
    atom2 = ligand.atoms[node2_idx]

    neighbour1 = []
    neighbour2 = []
    for neighbour_atom in openbabel.OBAtomAtomIter(atom1.OBAtom):
        if neighbour_atom.GetAtomicNum() != 1:
            neighbour1.append(neighbour_atom.GetIdx() -h_num[neighbour_atom.GetIdx()] - 1)

    for neighbour_atom in openbabel.OBAtomAtomIter(atom2.OBAtom):
        if neighbour_atom.GetAtomicNum() != 1:
            neighbour2.append(neighbour_atom.GetIdx() -h_num[neighbour_atom.GetIdx()] - 1)

    neighbour1.remove(node2_idx)
    neighbour2.remove(node1_idx)
    neighbour1.extend(neighbour2)

    angel_list = []
    area_list = []
    distence_list = []

    if len(neighbour1) == 0 and len(neighbour2) == 0:
        return [0, 0, 0, 0, 0, 0, 0, 0, 0]

    for node3_idx in neighbour1:
        node1_coord = np.array(ligand.atoms[node1_idx].coords)
        node2_coord = np.array(ligand.atoms[node2_idx].coords)
        node3_coord = np.array(ligand.atoms[node3_idx].coords)

        angel, area, distence = info_3D(node1_coord, node2_coord, node3_coord)
        angel_list.append(angel)
        area_list.append(area)
        distence_list.append(distence)

    for node3_idx in neighbour2:
        node1_coord = np.array(ligand.atoms[node1_idx].coords)
        node2_coord = np.array(ligand.atoms[node2_idx].coords)
        node3_coord = np.array(ligand.atoms[node3_idx].coords)
        angel, area, distence = info_3D(node2_coord, node1_coord, node3_coord)
        angel_list.append(angel)
        area_list.append(area)
        distence_list.append(distence)

    return [np.max(angel_list) * 0.01, np.sum(angel_list) * 0.01, np.mean(angel_list) * 0.01,
            np.max(area_list), np.sum(area_list), np.mean(area_list),
            np.max(distence_list) * 0.1, np.sum(distence_list) * 0.1, np.mean(distence_list) * 0.1]

def get_complex_edge_fea(edge_list,coord_list):

    net = nx.Graph()
    net.add_weighted_edges_from(edge_list)
    edges_fea = []
    for edge in edge_list:
        edge_fea = []
        edge_fea.append(edge[2])
        edges_fea.append(edge_fea)

    return edges_fea


def read_ligand(filepath):
    print('读取化合物')
    featurizer = Featurizer(save_molecule_codes=False)
    ligand = next(pybel.readfile("mol2", filepath))
    ligand_coord, atom_fea,h_num = featurizer.get_features(ligand)
    ligand_center = torch.tensor(ligand_coord).mean(dim=-2, keepdim=True)

    # smiles_fea = get_34Atomfea(filepath)
    smiles_fea = None
    return ligand_coord, atom_fea, smiles_fea, ligand, h_num, ligand_center

def get_prostt5_emb(model, tokenizer, seq):
    sequence_examples = [seq]
    sequence_examples = [" ".join(list(re.sub(r"[UZOB]", "X", seq))) for seq in sequence_examples]
    sequence_examples = [("<AA2fold>" + " " + s) if s.isupper() else ("<fold2AA>" + " " + s)
                         for s in sequence_examples]
    ids = tokenizer.batch_encode_plus(sequence_examples,
                                      add_special_tokens=True,
                                      padding="longest",
                                      return_tensors='pt').to(device)
    
    with torch.no_grad():
        embedding_rpr = model(ids.input_ids, attention_mask=ids.attention_mask)

    return embedding_rpr.last_hidden_state[0,2:len(sequence_examples[0])]

def read_protein(filepath, tokenizer, prot_lm):
    print('读取蛋白质')
    featurizer = Featurizer(save_molecule_codes=False)
    protein_pocket = next(pybel.readfile("pdb", filepath))
    pocket_coord, atom_fea,h_num = featurizer.get_features(protein_pocket)

    aa_codes = {
        'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E',
        'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 'LYS': 'K',
        'ILE': 'I', 'LEU': 'L', 'MET': 'M', 'ASN': 'N',
        'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S',
        'THR': 'T', 'VAL': 'V', 'TYR': 'Y', 'TRP': 'W'}
    env_seq = ''
    seq = ''
    protein_filepath = filepath.replace('_pocket','_protein')
    print(protein_filepath)
    for line in open(protein_filepath):
        if line[0:6] == "SEQRES":
            columns = line.split()
            for resname in columns[4:]:
                if resname in aa_codes:
                    # print(resname)
                    seq = seq + aa_codes[resname] + ' '
                    env_seq = env_seq + aa_codes[resname]
    # sequences_Example = re.sub(r"[UZOB]", "X", seq)

    # embedding = prot_lm(sequences_Example)
    # embedding = torch.tensor(embedding[0])
    # print(len(env_seq))
    # print(type(env_seq), env_seq)
    # embedding = get_prostt5_emb(prot_lm, tokenizer, env_seq)
    # pro_seq_emb = Data(seq = embedding)

    return pocket_coord, atom_fea,protein_pocket,h_num,None, env_seq

def Mult_graph(lig_file_name, pocket_file_name, id, score, tokenizer, port_lm):
    lig_coord, lig_atom_fea, smiles_fea, mol, h_num_lig, ligand_center = read_ligand(lig_file_name)
    pocket_coord, pocket_atom_fea,protein,h_num_pro,pro_seq, env_seq= read_protein(pocket_file_name, tokenizer, port_lm)  #,pro_seq

    if(mol != None) and (protein != None):
        G_l = Ligand_graph(lig_atom_fea, mol, h_num_lig,score)
        # lig = Data(seq=smiles_fea)
        G_inter = Inter_graph(lig_coord, pocket_coord, lig_atom_fea, pocket_atom_fea, env_seq, lig_file_name, score)
        
        pocket_coord = torch.tensor(pocket_coord, dtype=torch.float32)
        p_edge_index = torch_cluster.radius_graph(pocket_coord, r = 5.0)
        pocket_atom_fea = torch.tensor(pocket_atom_fea, dtype=torch.float32)
        G_p = Data(x=pocket_atom_fea, edge_index=p_edge_index, coords=pocket_coord)
        # G_list = [G_l, G_inter, pro_seq, id, ligand_center]
        # print(pro_seq.seq.shape)
        G_list = [G_l, G_inter]#
        return G_list, ligand_center, G_p
    else:
        return None

def bond_fea(bond,atom1,atom2):
    is_Aromatic = int(bond.IsAromatic())
    is_inring = int(bond.IsInRing())
    d = atom1.GetDistance(atom2)

    node1_idx = atom1.GetIdx()
    node2_idx = atom2.GetIdx()

    neighbour1 = []
    neighbour2 = []
    for neighbour_atom in openbabel.OBAtomAtomIter(atom1):
        if (neighbour_atom.GetAtomicNum() != 1 ) and (neighbour_atom.GetIdx() != node2_idx) :
            neighbour1.append(neighbour_atom)

    for neighbour_atom in openbabel.OBAtomAtomIter(atom2):
        if ( neighbour_atom.GetAtomicNum() != 1) and (neighbour_atom.GetIdx() != node1_idx):
            neighbour2.append(neighbour_atom)

    if len(neighbour1) == 0 and len(neighbour2) == 0:
        return [d,0, 0, 0, 0, 0, 0, 0, 0, 0,is_Aromatic,is_Aromatic]

    angel_list = []
    area_list = []
    distence_list = []

    node1_coord = np.array([atom1.GetX(),atom1.GetY(),atom1.GetZ()])
    node2_coord = np.array([atom2.GetX(),atom2.GetY(),atom2.GetZ()])

    for atom3 in neighbour1:
        node3_coord = np.array([atom3.GetX(), atom3.GetY(), atom3.GetZ()])
        angel, area, distence = info_3D(node1_coord, node2_coord, node3_coord)
        angel_list.append(angel)
        area_list.append(area)
        distence_list.append(distence)

    for atom3 in neighbour2:
        node3_coord = np.array([atom3.GetX(), atom3.GetY(), atom3.GetZ()])
        angel, area, distence = info_3D(node2_coord, node1_coord, node3_coord)
        angel_list.append(angel)
        area_list.append(area)
        distence_list.append(distence)

    return [d,
        np.max(angel_list) * 0.01, np.sum(angel_list) * 0.01, np.mean(angel_list) * 0.01,
        np.max(area_list), np.sum(area_list), np.mean(area_list),
        np.max(distence_list) * 0.1, np.sum(distence_list) * 0.1, np.mean(distence_list) * 0.1,
        is_Aromatic, is_inring]

def edgelist_to_tensor(edge_list):
    row = []
    column = []
    coo = []
    for edge in edge_list:
        row.append(edge[0])
        column.append(edge[1])

    coo.append(row)
    coo.append(column)

    coo = torch.Tensor(coo)
    edge_tensor = torch.tensor(coo, dtype=torch.long)
    return edge_tensor

def atomlist_to_tensor(atom_list):
    new_list = []
    for atom in atom_list:
        new_list.append([atom])
    atom_tensor = torch.Tensor(new_list)
    return atom_tensor

def Ligand_graph(lig_atoms_fea,ligand,h_num,score):
    edges = []
    edges_fea = []
    for bond in openbabel.OBMolBondIter(ligand.OBMol):
        atom1 = bond.GetBeginAtom()
        atom2 = bond.GetEndAtom()
        if (atom1.GetAtomicNum() == 1) or (atom2.GetAtomicNum() == 1):
            continue
        else:
            idx_1 = atom1.GetIdx() - h_num[atom1.GetIdx()-1] - 1
            idx_2 = atom2.GetIdx() - h_num[atom2.GetIdx()-1] - 1

            edge_fea = bond_fea(bond, atom1, atom2)
            edge = [idx_1, idx_2]
            edges.append(edge)
            edges_fea.append(edge_fea)

            re_edge = [idx_2, idx_1]
            edges.append(re_edge)
            edges_fea.append(edge_fea)

    edge_attr = torch.tensor(edges_fea, dtype=torch.float32)
    x = torch.tensor(lig_atoms_fea, dtype=torch.float32)
    edge_index = edgelist_to_tensor(edges)
    G_lig = Data(x=x, edge_attr=edge_attr, edge_index=edge_index, y=torch.tensor(score))

    return G_lig

# def normalize_tensor(tensor):
#     if isinstance(tensor, np.ndarray):
#         tensor = torch.tensor(tensor, dtype=torch.float32)
#     elif isinstance(tensor, list):
#         tensor = torch.tensor(tensor, dtype=torch.float32)

#     mean = tensor.mean(dim=0, keepdim=True)
#     std = tensor.std(dim=0, keepdim=True)
#     normalized_tensor = (tensor - mean) / (std + 1e-6)
#     return normalized_tensor

def getComplexEnv(sequence, mol):
    sequence = sequence.replace('X', '')
    protein_seq = Seq(sequence)
    # print(f'蛋白质sequence：{sequence}')
    analyzer = ProtParam.ProteinAnalysis(str(protein_seq))
    # molecular_weight = analyzer.molecular_weight()
    isoelectric_point = analyzer.isoelectric_point()
    hydropathy_index = analyzer.gravy()
    aromaticity = analyzer.aromaticity()
    charge_at_pH = analyzer.charge_at_pH(7.4)
    hb_donors = ['K', 'R', 'H', 'S', 'T', 'Y']
    hb_acceptors = ['E', 'D', 'N', 'Q', 'O']
    donor_count = sum(1 for aa in protein_seq if aa in hb_donors)
    acceptor_count = sum(1 for aa in protein_seq if aa in hb_acceptors)
    mol = Chem.MolFromMol2File(mol, removeHs=False)
    if mol is None:
        psa = 0
        logp = 0
        hb_donor = 0
        hb_acceptor = 0
        rotatable_bonds = 0
    else:
        psa = Descriptors.TPSA(mol)
        logp = Crippen.MolLogP(mol)
        hb_donor = Descriptors.NumHDonors(mol)
        hb_acceptor = Descriptors.NumHAcceptors(mol)
        rotatable_bonds = Descriptors.NumRotatableBonds(mol)

    # node1_features = [aromaticity, rotatable_bonds,
    #                   isoelectric_point, hydropathy_index, charge_at_pH, donor_count, acceptor_count,
    #                   psa, logp, hb_donor, hb_acceptor]
    # node_features = torch.tensor([node1_features], dtype=torch.float32)

    node1_features = [aromaticity, rotatable_bonds, 0, 0, 0]
    node2_features = [isoelectric_point, hydropathy_index, charge_at_pH, donor_count, acceptor_count]
    node3_features = [psa, logp, hb_donor, hb_acceptor, 0]
    node_features = torch.tensor([node1_features, node2_features, node3_features], dtype=torch.float32)
    # print(node_features)
    node_features = torch.nn.functional.normalize(node_features, p=2, dim=1)
    # print(node_features)
    return node_features

def compute_virtual_to_node_edges(env, fea):
    num_virtual_nodes = env.size(0)
    num_nodes = fea.size(0)
    source_nodes = torch.arange(num_virtual_nodes).repeat_interleave(num_nodes)
    target_nodes = torch.arange(num_nodes).repeat(num_virtual_nodes)
    return torch.stack([source_nodes, target_nodes], dim=0)

def cal_edge(coords, threshold = 10):
    coords = np.array(coords)
    dist_matrix = squareform(pdist(coords))
    edge_indices = np.array(np.nonzero((dist_matrix > 0) & (dist_matrix <= threshold))).T

    edge_distances = dist_matrix[dist_matrix > 0]
    edge_attr = dist_matrix[edge_indices[:, 0], edge_indices[:, 1]]

    edge_index = torch.tensor(edge_indices.T, dtype=torch.long)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float32).unsqueeze(-1)
    return edge_index, edge_attr


def Inter_graph(lig_coord, pocket_coord, lig_atom_fea, pocket_atom_fea, env_seq, lig_file_name, score, cut=5):
    print(f'化合物原子数量：{lig_coord.shape[0]}, 蛋白质原子数量：{pocket_coord.shape[0]}')
    coord_list = []
    for atom in lig_coord:
        coord_list.append(atom)
    for atom in pocket_coord:
        coord_list.append(atom)

    dis = distance_matrix(x=coord_list, y=coord_list)
    lenth = len(coord_list)
    edge_list = []

    edge_list_fea = []
    # Bipartite Graph; i belongs to ligand, j belongs to protein
    for i in range(len(lig_coord)):
        for j in range(len(lig_coord), lenth):
            if dis[i, j] < cut:
                edge_list.append([i, j-len(lig_coord), dis[i, j]])
                edge_list_fea.append([i,j,dis[i,j]])
    env = getComplexEnv(env_seq, lig_file_name)
    data = HeteroData()
    edge_index = edgelist_to_tensor(edge_list)

    data['ligand'].x = torch.tensor(lig_atom_fea, dtype=torch.float32)
    data['ligand'].y = torch.tensor(score)
    data['protein'].x = torch.tensor(pocket_atom_fea, dtype=torch.float32)
    data['ligand', 'protein'].edge_index = edge_index

    complex_edges_fea = get_complex_edge_fea(edge_list_fea,coord_list)
    edge_attr = torch.tensor(complex_edges_fea, dtype=torch.float32)
    data['ligand', 'protein'].edge_attr = edge_attr

    data['env'].x = env
    env2l_edge_index = compute_virtual_to_node_edges(env, data['ligand'].x)
    env2p_edge_index = compute_virtual_to_node_edges(env, data['protein'].x)
    data['env', 'protein'].edge_index = env2p_edge_index
    data['env', 'ligand'].edge_index = env2l_edge_index
    data['env', 'protein'].edge_attr = torch.zeros(env2p_edge_index.shape[1], dtype=torch.float32).unsqueeze(-1)
    data['env', 'ligand'].edge_attr = torch.zeros(env2l_edge_index.shape[1], dtype=torch.float32).unsqueeze(-1)

    data = T.ToUndirected()(data)

    return data

def get_Resfea(res):
    aa_codes = {
        'ALA': 1, 'CYS': 2, 'ASP': 3, 'GLU': 4,
        'PHE': 5, 'GLY': 6, 'HIS': 7, 'LYS': 8,
        'ILE': 9, 'LEU': 10, 'MET': 11, 'ASN': 12,
        'PRO': 13, 'GLN': 14, 'ARG': 15, 'SER': 16,
        'THR': 17, 'VAL': 18, 'TYR': 19, 'TRP': 0}
    one_hot = np.eye(21)
    if res in aa_codes:
        code = aa_codes[res]
    else:
        code = 20
    fea = one_hot[code]
    return fea

def GetPDBDict(Path):
    with open(Path, 'rb') as f:
        lines = f.read().decode().strip().split('\n')
    res = {}
    for line in lines:
        if "//" in line:
            temp = line.split()
            name, score = temp[0], float(temp[3])
            res[name] = score
    return res

def load_structure_np(fname):
    # Load the data
    parser = PDBParser()
    structure = parser.get_structure("structure", fname)
    atoms = structure.get_atoms()

    coords = []
    types = []
    for atom in atoms:
        coords.append(atom.get_coord())
        types.append(ele2num[atom.element])

    coords = np.stack(coords)
    types_array = np.zeros((len(types), 22))
    for i, t in enumerate(types):
        types_array[i, t] = 1.0

    return coords, types_array

def load_protein_atoms(protein_path, item):
    try:
        atom_coords, atom_types = load_structure_np(protein_path+item+"_protein.pdb")
    except:
        return None

    protein_data = Data(
        atom_coords=torch.tensor(atom_coords),
        atom_types=torch.tensor(atom_types,dtype=torch.float32),
    )
    return protein_data

def pad_or_truncate_1d(tensor, target_len, pad_value=0):
    current_len = tensor.shape[0]
    if current_len < target_len:
        pad_size = target_len - current_len
        tensor = torch.cat([tensor, torch.full((pad_size, tensor.shape[1]), pad_value)], dim=0)
        # mask = torch.cat([torch.ones(current_len), torch.zeros(pad_size)], dim=0)
    else:
        tensor = tensor[:target_len]
        # mask = torch.ones(target_len)
    return tensor#, mask

def pad_or_truncate_2d(tensor, target_len, pad_value=0):
    current_len = tensor.shape[0]
    if current_len < target_len:
        pad_size = target_len - current_len
        row_pad = torch.full((pad_size, tensor.shape[1]), pad_value)
        tensor = torch.cat([tensor, row_pad], dim=0)
        col_pad = torch.full((tensor.shape[0], pad_size), pad_value)
        tensor = torch.cat([tensor, col_pad], dim=1)
    elif current_len > target_len:
        tensor = tensor[:target_len, :target_len]
    return tensor

def cal_dssp(pdb_file):
    aa_to_onehot = {letter: np.eye(21)[i] for i, letter in enumerate('ACDEFGHIKLMNPQRSTVWYX')}
    secStr_to_onehot = {letter: np.eye(8)[i] for i, letter in enumerate('GHIEBTSC')}
    secStr_to_onehot['-'] = np.zeros(8)
    property_encoding = {
        'C': [1, 0, 0, 0],  # Non-polar
        'D': [0, 1, 1, 0],  # Acidic
        'E': [0, 1, 1, 0],  # Acidic
        'R': [0, 0, 0, 1],  # Basic
        'K': [0, 0, 0, 1],  # Basic
        'H': [0, 0, 0, 1],  # Basic
        'N': [0, 1, 0, 0],  # Polar
        'Q': [0, 1, 0, 0],  # Polar
        'W': [1, 0, 0, 0],  # Non-polar
        'Y': [1, 0, 0, 0],  # Non-polar
        'M': [1, 0, 0, 0],  # Non-polar
        'T': [0, 1, 0, 0],  # Polar
        'S': [0, 1, 0, 0],  # Polar
        'I': [1, 0, 0, 0],  # Non-polar
        'L': [1, 0, 0, 0],  # Non-polar
        'F': [1, 0, 0, 0],  # Non-polar
        'P': [1, 0, 0, 0],  # Non-polar
        'A': [1, 0, 0, 0],  # Non-polar
        'G': [1, 0, 0, 0],  # Non-polar
        'V': [1, 0, 0, 0],  # Non-polar
        'X': [0, 0, 0, 0],
    }
    cluster_assignment = {
        'C': [1, 0, 0, 0, 0, 0, 0],  # Cluster 1
        'D': [0, 1, 0, 0, 0, 0, 0],  # Cluster 2
        'E': [0, 1, 0, 0, 0, 0, 0],  # Cluster 2
        'R': [0, 0, 1, 0, 0, 0, 0],  # Cluster 3
        'K': [0, 0, 1, 0, 0, 0, 0],  # Cluster 3
        'H': [0, 0, 0, 1, 0, 0, 0],  # Cluster 4
        'N': [0, 0, 0, 1, 0, 0, 0],  # Cluster 4
        'Q': [0, 0, 0, 1, 0, 0, 0],  # Cluster 4
        'W': [0, 0, 0, 1, 0, 0, 0],  # Cluster 4
        'Y': [0, 0, 0, 0, 1, 0, 0],  # Cluster 5
        'M': [0, 0, 0, 0, 1, 0, 0],  # Cluster 5
        'T': [0, 0, 0, 0, 1, 0, 0],  # Cluster 5
        'S': [0, 0, 0, 0, 1, 0, 0],  # Cluster 5
        'I': [0, 0, 0, 0, 0, 1, 0],  # Cluster 6
        'L': [0, 0, 0, 0, 0, 1, 0],  # Cluster 6
        'F': [0, 0, 0, 0, 0, 1, 0],  # Cluster 6
        'P': [0, 0, 0, 0, 0, 1, 0],  # Cluster 6
        'A': [0, 0, 0, 0, 0, 0, 1],  # Cluster 7
        'G': [0, 0, 0, 0, 0, 0, 1],  # Cluster 7
        'V': [0, 0, 0, 0, 0, 0, 1],  # Cluster 7
        'X': [0, 0, 0, 0, 0, 0, 0],
    }
    
    structure = PDB.PDBParser(QUIET=True).get_structure('protein', pdb_file)
    try:
        feature = []
        dssp = PDB.DSSP(structure[0], pdb_file)
        for res in dssp:
            amino_acid = res[1] 
            sec_structure = res[2] 
            f = aa_to_onehot[amino_acid].tolist()
            f.extend(secStr_to_onehot[sec_structure].tolist())
            f.extend(property_encoding[amino_acid])
            f.extend(cluster_assignment[amino_acid])
            feature.append(f)
        print('dssp')
    except:
        feature = []
        for chain in structure[0]:
            for residue in chain:
                if not residue.id[0] == " ":
                    continue

                resname = residue.get_resname()
                if resname not in aa_to_onehot:
                    resname = 'X'
                sec_structure = "-"
                f = aa_to_onehot[resname].tolist()
                f.extend(secStr_to_onehot[sec_structure].tolist())
                f.extend(property_encoding[resname])
                f.extend(cluster_assignment[resname])
                feature.append(f)
        print('pdb')
    feature = torch.tensor(feature, dtype=torch.float)
    # print(feature.shape)
    return feature[:1024]

def load_protein_map(protein_path, item, ligand_center, dssp_fea, tokenizer, prot_lm):
    def get_nearest_coords(seq, coords, ligand_center, top_k=512):
        diffs = coords - ligand_center
        distances = torch.norm(diffs, dim=1)
        top_k = min(top_k, coords.size(0))
        _, indices = torch.topk(distances, top_k, largest=False)
        nearest_seq = ''.join([seq[i] for i in indices])
        nearest_coords = coords[indices]
        return nearest_seq, nearest_coords
    aa_codes = {
        'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E',
        'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 'LYS': 'K',
        'ILE': 'I', 'LEU': 'L', 'MET': 'M', 'ASN': 'N',
        'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S',
        'THR': 'T', 'VAL': 'V', 'TYR': 'Y', 'TRP': 'W',
        }
    aa_to_onehot = {letter: np.eye(21)[i] for i, letter in enumerate('ACDEFGHIKLMNPQRSTVWYX')}

    path = protein_path+item+"_protein.pdb"
    protein = fo.bp_to_df(fo.read_pdb(path))

    
    # protein_graph = featurize_as_graph(protein_df)
    name = protein['name']
    N_coords = protein[protein.name == 'N'][['x', 'y', 'z']].to_numpy()
    CA_coords =protein[protein.name == 'CA'][['x', 'y', 'z']].to_numpy()
    C_coords = protein[protein.name == 'C'][['x', 'y', 'z']].to_numpy()
    O_coords = protein[protein.name == 'O'][['x', 'y', 'z']].to_numpy()
    max_ = min([CA_coords.shape[0], C_coords.shape[0], N_coords.shape[0], O_coords.shape[0]])
    N_coords = N_coords[:max_, :]
    CA_coords = CA_coords[:max_, :]
    C_coords = C_coords[:max_, :]
    O_coords = O_coords[:max_, :]
    coords = np.stack((N_coords, CA_coords,C_coords, O_coords), axis = 1)
    with torch.no_grad():
        coords = torch.from_numpy(coords).float()
        seq = [aa_codes.get(a, 'X') for a in protein[protein.name == 'CA']['resname'][:max_]]
        seq = ''.join(seq)
        coords = coords[:, 1]

        seq_all, all_coords = get_nearest_coords(seq, coords, ligand_center, top_k=512)
        seq_part, part_coords = get_nearest_coords(seq, coords, ligand_center, top_k=64)#128/64

        seq_cnn = seq_all
        seq_cnn = np.array([aa_to_onehot[aa] for aa in seq_cnn])
        seq_cnn_all = torch.tensor(seq_cnn, dtype=torch.float)

        # print(name)
        try:
            d_fea = dssp_fea[item]
            seq_sec_all = d_fea[0]
            seq_sec_part = d_fea[1]
        except:
            seq_sec_all = cal_dssp(protein_path+item+"_protein.pdb")
            seq_sec_part = cal_dssp(protein_path+item+"_pocket.pdb")
        print(seq_cnn_all.shape, seq_sec_all.shape)

        # embedding_all = get_prostt5_emb(prot_lm, tokenizer, seq_all).to('cpu')
        # embedding_part = get_prostt5_emb(prot_lm, tokenizer, seq_part).to('cpu')

        # coords = part_coords
        # coords_expanded = coords.unsqueeze(0) 
        # coords_transposed = coords.unsqueeze(1) 
        # diff = coords_expanded - coords_transposed
        # dis_map_part = torch.sqrt(torch.sum(diff**2, dim=2))

        # coords = all_coords
        # coords_expanded = coords.unsqueeze(0) 
        # coords_transposed = coords.unsqueeze(1) 
        # diff = coords_expanded - coords_transposed
        # dis_map_all = torch.sqrt(torch.sum(diff**2, dim=2))

        # if(part_coords.shape[0]<128):
        #     print("填充64")
        #     embedding_part = pad_or_truncate_1d(embedding_part, target_len=128)
            # dis_map_part = pad_or_truncate_2d(dis_map_part, target_len=128)
        
        # if(all_coords.shape[0]<512):
        #     print("512")
        #     dis_map_all = pad_or_truncate_2d(dis_map_all, target_len=512)
        # pro_all = Data(seq=embedding_all, seq_cnn=seq_cnn_all, coord=all_coords)
        # pro_part = Data(seq=embedding_part, coord=part_coords)
        pro_all = None
        pro_part = None
        pro_dssp_all = Data(seq=seq_sec_all)
        pro_dssp_part = Data(seq=seq_sec_part)

    return pro_all, pro_part, pro_dssp_all, pro_dssp_part
    # return embedding_padded, distances_padded, seq_mask

def load_protein_graph(protein_path, item):

    path = protein_path+item+"_pocket.pdb"
    # protein_graph = featurize_as_graph_dssp(path)
    protein_df = fo.bp_to_df(fo.read_pdb(path))
    protein_graph = featurize_as_graph(protein_df)

    return protein_graph

def process_raw_data(dataset_path, processed_file, protein_list):
    res = GetPDBDict(Path='./data/PdbBind/index/INDEX_general_PL_data.2016')

    G_list = []
    # tokenizer = T5Tokenizer.from_pretrained('prostt5/', do_lower_case=False)
    # model = T5EncoderModel.from_pretrained("prostt5/").to(device)
    tokenizer = None
    model = None

    f_dssp = open('dssp.pkl', 'rb')
    dssp_fea = pickle.load(f_dssp)
    
    # print(dssp_fea.)
    # tokenizer = BertTokenizer.from_pretrained("probert", do_lower_case=False )
    # model = BertModel.from_pretrained("probert")
    # fe = pipeline('feature-extraction', model=model, tokenizer=tokenizer,device=0)
    # model = fe
    all_num=0
    suc_num=0
    has_item = []
    for item in tqdm(protein_list):
        # print(f'当前：{item}')
        score = res[item]
        lig_file_name = dataset_path + item + '/' + item + '_ligand.mol2'
        pocket_file_name = dataset_path + item + '/' + item + '_pocket.pdb'
        protein_path = dataset_path + item  + '/'
        if not os.path.exists(protein_path):
            continue
        if(item in has_item):
            continue
        # all_num+=1
        # print(item)
        G=[]
        G, ligand_center, _ = Mult_graph(lig_file_name, pocket_file_name, item, score, tokenizer, model)
        # protein_atoms = load_protein_atoms(protein_path, item)
        pro_all, pro_part, pro_dssp_all, pro_dssp_part = load_protein_map(protein_path, item, ligand_center, dssp_fea, tokenizer, model)
        protein_graph = load_protein_graph(protein_path,item)
        if(G != None):
            print(f'成功生成：{item}')
            # suc_num+=1
            has_item.append(item)
            # G.append(pro_all)
            # G.append(pro_part)
            G.append(pro_dssp_all)
            G.append(pro_dssp_part)
            G.append(protein_graph)
            G_list.append(G)
    # print(all_num, suc_num)
    print('sample num: ', len(G_list))
    with open(processed_file, 'wb') as f:
        pickle.dump(G_list, f)
    f.close()

def GetPDBList(Path):
    with open(Path, 'rb') as f:
        lines = f.read().decode().strip().split('\n')
    res = []
    for line in lines:
        if "//" in line:
            temp = line.split()
            res.append(temp[0])
    return res


if __name__ == '__main__':
    dataset = 'csar_87'#refined-set #CSAR-HIQ_51
    raw_data_path = f'./data/PdbBind/{dataset}/'
    protein_list = GetPDBList(Path='./data/PdbBind/index/INDEX_general_PL_data.2016')

    process_raw_data(raw_data_path, f'./data/{dataset}.pkl', protein_list)

    
# huggingface-cli download --resume-download Rostlab/prot_bert --local-dir probert
# huggingface-cli download --resume-download Rostlab/ProstT5 --local-dir ProstT5