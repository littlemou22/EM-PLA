# import os
# import os
# import torch
# # import openbabel
# from openbabel import pybel
# from utils.openbabel_featurizer import Featurizer
# import numpy as np
# from Bio.PDB import PDBParser, PDBIO, Select

# class ResiduePocketSelect(Select):
#     def __init__(self, center, cutoff):
#         self.center = np.array(center)
#         self.cutoff = cutoff

#     def accept_residue(self, residue):
#         # 获取氨基酸中所有原子的坐标
#         atom_coords = [atom.coord for atom in residue if atom.coord is not None]
#         if not atom_coords:  # 如果没有原子，跳过
#             return False
        
#         # 计算所有原子坐标的均值
#         residue_center = np.mean(atom_coords, axis=0)
        
#         # 计算均值与 ligand_center 的距离
#         distance = np.linalg.norm(residue_center - self.center)
        
#         # 如果距离小于等于 cutoff，将整个氨基酸所有原子加入
#         return distance <= self.cutoff
# def write_seqres_to_file(pdb_file_path):
#     residues_by_chain = {}

#     # 读取PDB文件内容
#     with open(pdb_file_path, 'r') as file:
#         pdb_content = file.readlines()

#     # 提取不同链的氨基酸序号和名称
#     for line in pdb_content:
#         if line.startswith("ATOM"):
#             chain_id = line[21].strip()  # 链 ID
#             res_seq = int(line[22:26].strip())  # 氨基酸序号
#             res_name = line[17:20].strip()     # 氨基酸名称
#             if chain_id not in residues_by_chain:
#                 residues_by_chain[chain_id] = {}
#             if res_seq not in residues_by_chain[chain_id]:
#                 residues_by_chain[chain_id][res_seq] = res_name

#     # 生成每条链的 SEQRES 内容
#     seqres_lines = []
#     for chain_id, residues in residues_by_chain.items():
#         # 按序号排序并提取氨基酸名称
#         sorted_residues = [residues[key] for key in sorted(residues.keys())]
#         total_residues = len(sorted_residues)

#         # 按SEQRES格式分行
#         line_number = 1
#         for i in range(0, len(sorted_residues), 13):  # 每行最多13个氨基酸
#             seqres_line = f"SEQRES   {line_number:<2} {chain_id}  {total_residues:<4}  {' '.join(sorted_residues[i:i+13])}"
#             seqres_lines.append(seqres_line)
#             line_number += 1

#     # 将SEQRES内容和原PDB内容组合
#     updated_content = "\n".join(seqres_lines) + "\n" + "".join(pdb_content)

#     # 写回同名文件
#     with open(pdb_file_path, 'w') as file:
#         file.write(updated_content)

    

# # 示例用法
# dataset = 'csar_test'
# for file in os.listdir(f'data/PdbBind/{dataset}'):
#     if(not os.path.exists(f"data/PdbBind/{dataset}/{file}/{file}_valid_pocket.pdb")):
#         print(file)
#     featurizer = Featurizer(save_molecule_codes=False)
#     ligand = next(pybel.readfile("mol2", f"data/PdbBind/{dataset}/{file}/{file}_ligand.mol2"))
#     ligand_coord, atom_fea, h_num = featurizer.get_features(ligand)
#     ligand_center = torch.tensor(ligand_coord).mean(dim=-2, keepdim=True)
    
#     ligand_center = np.array(ligand_center)  # 替换x, y, z为具体坐标
#     parser = PDBParser(QUIET=True)
#     structure = parser.get_structure("protein", f"data/PdbBind/{dataset}/{file}/{file}_protein.pdb")

#     # 定义输出文件名
#     output_file = "pocket.pdb"

#     # 设置距离阈值
#     distance_cutoff = 12.0

#     # 保存结果
#     io = PDBIO()
#     io.set_structure(structure)
#     io.save(output_file, select=ResiduePocketSelect(center=ligand_center, cutoff=distance_cutoff))


#     break

import os
import shutil

# 定义文件夹路径
csar_test_path = 'data/PdbBind/csar_test'
core_set_path = 'data/PdbBind/core-set'
refined_set_path = 'data/PdbBind/refined-set'
general_set_path = 'data/PdbBind/general-set'
csar_135_path = 'data/PdbBind/csar_135'

# 创建csar_135文件夹（如果不存在）
if not os.path.exists(csar_135_path):
    os.makedirs(csar_135_path)

# 获取core-set文件夹中的所有文件夹名称
core_set_folders = set(os.listdir(core_set_path))
refined_set_folders = set(os.listdir(refined_set_path))
general_set_folders = set(os.listdir(general_set_path))
n=0
# 遍历csar_test文件夹中的所有文件夹
for folder in os.listdir(csar_test_path):
    folder_path = os.path.join(csar_test_path, folder)
    
    # 检查该项是否为文件夹
    if os.path.isdir(folder_path):
        # 如果该文件夹不在core-set中，则移动该文件夹到csar_135
        if folder not in refined_set_folders:#folder not in core_set_folders and folder not in refined_set_folders and folder not in general_set_folders:
            destination_path = os.path.join(csar_135_path, folder)
            
            # 移动文件夹及其所有内容
            shutil.copytree(folder_path, destination_path)
            print(f"Moved {folder} to {csar_135_path}")
            n+=1
            print(n)