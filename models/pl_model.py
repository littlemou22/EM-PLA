import torch
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, Linear, MLP, SAGEConv
from torch_scatter import scatter_sum, scatter_max
from torch_geometric.utils import to_dense_batch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.manifold import TSNE
from matplotlib.colors import Normalize
n=0
n2=0
# def plot_attention(x_data, y_data):
#     print(x_data.shape, y_data.shape)
#     attention_scores = torch.matmul(x_data, y_data.T)  # 点积相似度
#     attention_weights = F.softmax(attention_scores, dim=-1)  # 在最后一个维度进行归一化
#     print(attention_weights.shape)
#     x_nodes = torch.arange(x_data.size(0))
#     y_nodes = torch.arange(y_data.size(0))
#     # 注意力权重矩阵的形状
#     fig, ax = plt.subplots(figsize=(10, 8))
    
#     sns.heatmap(attention_weights.detach().cpu().numpy(), cmap='Blues', cbar=True, ax=ax,
#                 xticklabels=y_data, yticklabels=x_data)
    
#     ax.set_xlabel('Protein Nodes')
#     ax.set_ylabel('Ligand Nodes')
#     ax.set_title('Attention Map between Ligand and Protein')
#     plt.savefig('attention.png')
#     # plt.tight_layout()
#     # plt.show()
#     return fig

def plot_edge(edge_index_dict, num_ligand_nodes, num_protein_nodes):
    global n
    global n2
    # 创建一个空白的邻接矩阵
    adj_matrix = np.zeros((num_ligand_nodes, num_protein_nodes))

    # 根据边的信息填充邻接矩阵
    
    ligand_nodes = edge_index_dict[('ligand', 'to', 'protein')][0]
    protein_nodes = edge_index_dict[('ligand', 'to', 'protein')][1]

    for ligand_node, protein_node in zip(ligand_nodes, protein_nodes):
        adj_matrix[ligand_node, protein_node] = 1  # 如果有边，设置为 0（深蓝色）

    # 绘制图像
    plt.figure(figsize=(10, 8))
    plt.imshow(adj_matrix, cmap='Blues', interpolation='nearest')
    plt.colorbar(label="Edge Existence")

    # 设置标签
    plt.xlabel("Protein Node", fontsize=12)
    plt.ylabel("Ligand Node", fontsize=12)
    plt.title("Ligand to Protein Connection", fontsize=14)

    # 设置横坐标和纵坐标的刻度
    plt.xticks(np.arange(num_protein_nodes), labels=np.arange(num_protein_nodes))
    plt.yticks(np.arange(num_ligand_nodes), labels=np.arange(num_ligand_nodes))

    # 显示图像
    plt.savefig(f'attention_map/edge/map_{n}.png', bbox_inches='tight')
    print(f"Optimized image saved as 'weisfeiler_lehman_map' folder")

    # Close the figure to free memory
    # plt.close(fig)
# import torch
# import torch.nn.functional as F
# import numpy as np
# import matplotlib.pyplot as plt

def plot_weisfeiler_lehman(x_data, y_data, x_mask, y_mask, mode='lig', num_iterations=5, edge_index_dict=None, edge_attr_dict=None):
    # 计算edge，距离要小于1
    print('lehman')
    # global n
    # global n2
    
    # Apply masks to filter valid nodes
    x_data = x_data[x_mask]  # Keep only valid x nodes
    y_data = y_data[y_mask]  # Keep only valid y nodes
    
    # Perform the Weisfeiler-Leman test on the x_data and y_data
    x_updated = x_data
    y_updated = y_data
    
    # Compute the similarity between updated x_data and y_data (dot product for simplicity)
    similarity_scores = torch.matmul(x_updated, y_updated.T)  # Similarity based on updated features
    similarity_scores = F.softmax(similarity_scores, dim=-1)  # Normalize along the last dimension

    # Create figure and axis for plotting
    fig, ax = plt.subplots(figsize=(12, 10))  # Increase image size
    
    # Use imshow to plot the similarity matrix (this can also be used for a heatmap)
    cax = ax.imshow(similarity_scores.detach().cpu().numpy(), cmap='Blues', interpolation='none', vmax=0.01, vmin=0)

    # Add colorbar
    fig.colorbar(cax, ax=ax, orientation='vertical')

    # Set axis labels and title
    ax.set_xlabel('Protein Nodes')
    ax.set_ylabel('Ligand Nodes')
    ax.set_title(f'Weisfeiler-Leman Test Similarity Map')

    # Adjust ticks to display part of the axis labels
    step_x = max(1, x_data.shape[0] // 10)
    step_y = max(1, y_data.shape[0] // 10)
    ax.set_xticks(np.arange(0, y_data.shape[0], step_y))
    ax.set_xticklabels(np.arange(0, y_data.shape[0], step_y), fontsize=8, rotation=45)
    ax.set_yticks(np.arange(0, x_data.shape[0], step_x))
    ax.set_yticklabels(np.arange(0, x_data.shape[0], step_x), fontsize=8)

    # If edge_index_dict is provided, draw the edges on the heatmap
    if edge_index_dict is not None:
        ligand_nodes = edge_index_dict[('ligand', 'to', 'protein')][0]
        protein_nodes = edge_index_dict[('ligand', 'to', 'protein')][1]

        print(edge_attr_dict.keys())
        dis = edge_attr_dict[('ligand', 'to', 'protein')]
        
        for ligand_node, protein_node, d in zip(ligand_nodes, protein_nodes, dis):
            # print(d)
            if(d<3.5):
                ax.plot(protein_node.cpu(), ligand_node.cpu(), 'ro', markersize=0.5)  # Draw red dots for edges
    
    # Save the image
    plt.savefig(f'attention_map/lehman/{mode}_map.png', bbox_inches='tight')
    print(f"Optimized image saved as 'weisfeiler_lehman_map' folder")

    # Close the figure to free memory
    plt.close(fig)

    return fig

# def plot_weisfeiler_lehman(x_data, y_data, x_mask, y_mask, mode='lig', num_iterations=5):
#     print('lehman')
#     global n
#     global n2
    
#     # Apply masks to filter valid nodes
#     x_data = x_data[x_mask]  # Keep only valid x nodes
#     y_data = y_data[y_mask]  # Keep only valid y nodes
    
#     # print(f"Filtered shapes: x_data={x_data.shape}, y_data={y_data.shape}")

#     # Perform the Weisfeiler-Leman test on the x_data and y_data
#     x_updated=x_data
#     y_updated=y_data
#     # = weisfeiler_lehman(x_data, y_data, num_iterations)
    
#     # Compute the similarity between updated x_data and y_data (dot product for simplicity)
#     similarity_scores = torch.matmul(x_updated, y_updated.T)  # Similarity based on updated features
#     similarity_scores = F.softmax(similarity_scores, dim=-1)  # 在最后一个维度进行归一化

#     # Create figure and axis for plotting
#     fig, ax = plt.subplots(figsize=(12, 10))  # Increase image size
    
#     # Use imshow to plot the similarity matrix (this can also be used for a heatmap)
#     cax = ax.imshow(similarity_scores.detach().cpu().numpy(), cmap='Blues', interpolation='none', vmax=0.01, vmin=0)

#     # Add colorbar
#     fig.colorbar(cax, ax=ax, orientation='vertical')

#     # Set axis labels and title
#     ax.set_xlabel('Protein Nodes')
#     ax.set_ylabel('Ligand Nodes')
#     ax.set_title(f'Weisfeiler-Leman Test Similarity Map')

#     # Adjust ticks to display part of the axis labels
#     step_x = max(1, x_data.shape[0] // 10)
#     step_y = max(1, y_data.shape[0] // 10)
#     ax.set_xticks(np.arange(0, y_data.shape[0], step_y))
#     ax.set_xticklabels(np.arange(0, y_data.shape[0], step_y), fontsize=8, rotation=45)
#     ax.set_yticks(np.arange(0, x_data.shape[0], step_x))
#     ax.set_yticklabels(np.arange(0, x_data.shape[0], step_x), fontsize=8)

#     # Save the image
#     plt.savefig(f'attention_map/lehman/{mode}_map_{n}.png', bbox_inches='tight')
#     print(f"Optimized image saved as 'weisfeiler_lehman_map' folder")

#     # Close the figure to free memory
#     plt.close(fig)

#     return fig

# def plot_tsne(x_data, y_data, x_mask, y_mask, mode='lig'):
#     print('tsne')
#     global n
#     global n2
#     # Filter x_data and y_data using the provided masks
#     x_data = x_data[x_mask]  # Filter valid x nodes
#     y_data = y_data[y_mask]  # Filter valid y nodes

#     # print(f"Filtered shapes: x_data={x_data.shape}, y_data={y_data.shape}")

#     # Convert x_data and y_data to numpy arrays for t-SNE
#     x_data_np = x_data.detach().cpu().numpy()  # Convert to NumPy (for t-SNE)
#     y_data_np = y_data.detach().cpu().numpy()  # Convert to NumPy (for t-SNE)

#     # Combine x_data and y_data for t-SNE (for joint visualization)
#     combined_data = np.vstack([x_data_np, y_data_np])

#     # Apply t-SNE to reduce dimensionality to 2D
#     tsne = TSNE(n_components=2, random_state=42)
#     reduced_data = tsne.fit_transform(combined_data)

#     # Split back the reduced data into x_data and y_data components
#     x_reduced = reduced_data[:x_data_np.shape[0]]
#     y_reduced = reduced_data[x_data_np.shape[0]:]

#     # Plotting
#     plt.figure(figsize=(10, 8))

#     # Plot x_data points (protein nodes) in one color
#     plt.scatter(x_reduced[:, 0], x_reduced[:, 1], c='blue', label='Ligand Nodes', alpha=0.6)
    
#     # Plot y_data points (ligand or environment nodes) in another color
#     plt.scatter(y_reduced[:, 0], y_reduced[:, 1], c='red', label='Protein Nodes', alpha=0.6)

#     # Set labels and title
#     plt.xlabel('t-SNE Component 1')
#     plt.ylabel('t-SNE Component 2')
#     plt.title('t-SNE visualization of x_data and y_data')

#     # Add legend
#     plt.legend()

#     # Save the plot
#     # if mode == 'lig':
#     plt.savefig(f'attention_map/tsne/{mode}_map_{n}.png', bbox_inches='tight')
#     #     n += 1
#     # else:
#     #     plt.savefig(f'attention_map/tsne/map_tsne_{n2}.png', bbox_inches='tight')
#     #     n2 += 1

#     print(f"t-SNE plot saved as 'tsne_map' folder")


from sklearn.manifold import TSNE
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
def plot_tsne_with_attention(edge_index_dict, x_data, y_data, x_mask, y_mask, mode='lig', min_dist=0, max_dist=0.3):
    print('t-SNE with distance-based connection, normalization, and edge coloring')

    ligand_nodes = edge_index_dict[('ligand', 'to', 'protein')][0]
    protein_nodes = edge_index_dict[('ligand', 'to', 'protein')][1]

    num_ligand_nodes = x_data[x_mask].shape[0]
    num_protein_nodes = y_data[y_mask].shape[0]
    adj_matrix = np.zeros((num_ligand_nodes, num_protein_nodes))
    for ligand_node, protein_node in zip(ligand_nodes, protein_nodes):
        adj_matrix[ligand_node, protein_node] = 1  # 如果有边，设置为 0（深蓝色）

    # Filter x_data and y_data using the provided masks
    x_data = x_data[x_mask]  # Filter valid x nodes
    y_data = y_data[y_mask]  # Filter valid y nodes

    # Convert x_data and y_data to numpy arrays for t-SNE
    x_data_np = x_data.detach().cpu().numpy()  # Convert to NumPy (for t-SNE)
    y_data_np = y_data.detach().cpu().numpy()  # Convert to NumPy (for t-SNE)

    # Combine x_data and y_data for t-SNE (for joint visualization)
    combined_data = np.vstack([x_data_np, y_data_np])

    # Apply t-SNE to reduce dimensionality to 3D
    tsne = TSNE(n_components=3, random_state=42)
    reduced_data = tsne.fit_transform(combined_data)

    # Normalize the t-SNE reduced data using Min-Max normalization
    scaler = MinMaxScaler()
    reduced_data_normalized = scaler.fit_transform(reduced_data)  # Normalize the data to [0, 1]

    # Split back the normalized reduced data into x_data and y_data components
    x_reduced = reduced_data_normalized[:x_data_np.shape[0]]
    y_reduced = reduced_data_normalized[x_data_np.shape[0]:]

    # Calculate all distances between points in the 3D space
    distances = []
    for i in range(x_reduced.shape[0]):
        for j in range(y_reduced.shape[0]):
            distance = np.linalg.norm(x_reduced[i] - y_reduced[j])  # Euclidean distance in 3D space
            distances.append((i, j, distance))

    # Sort distances by the smallest distance and select top 20%
    distances_sorted = sorted(distances, key=lambda x: x[2])  # Sort by distance (ascending)
    top_20_percent_count = int(len(distances_sorted) * 0.01)  # Get the top 20%

    # Plotting the histogram of distances
    sorted_distances = [dist[2] for dist in distances_sorted]
    plt.figure(figsize=(8, 6))
    plt.hist(sorted_distances, bins=50, color='skyblue', edgecolor='black')
    plt.xlim(0, 1.25)  # Set x-axis limits
    plt.ylim(0, 300)  # Set y-axis limits
    plt.title('Histogram of Sorted Distances Between Ligand and Protein Nodes')
    plt.xlabel('Distance (Euclidean)')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig(f'attention_map/tsne/{mode}_distance_histogram.png', bbox_inches='tight')
    print(f"Distance histogram saved.")

    # Prepare color mapping for edges (using the distance as the color scale)
    distance_values = [dist[2] for dist in distances_sorted[:top_20_percent_count]]  # Get distances of top 20%
    
    # Use the provided min_dist and max_dist for the colorbar range
    norm = Normalize(vmin=min_dist, vmax=max_dist)  # Normalize distances for colormap
    cmap = plt.cm.viridis  # Choose a colormap

    # Plotting
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot x_data points (ligand nodes) in one color
    ax.scatter(x_reduced[:, 0], x_reduced[:, 1], x_reduced[:, 2], c='blue', label='Ligand Nodes', alpha=0.6)

    # Plot y_data points (protein nodes) in another color
    ax.scatter(y_reduced[:, 0], y_reduced[:, 1], y_reduced[:, 2], c='red', label='Protein Nodes', alpha=0.6)

    # Plot lines for the top 20% closest points, with colors based on distance
    for i, j, distance in distances_sorted[:top_20_percent_count]:
        if distance < 0.2:
            color = cmap(norm(distance))  # Get color based on the distance
            ax.plot([x_reduced[i, 0], y_reduced[j, 0]], 
                    [x_reduced[i, 1], y_reduced[j, 1]], 
                    [x_reduced[i, 2], y_reduced[j, 2]], 
                    c=color, alpha=0.5)  # Line between closest points with color

    # Add edges from edge_index_dict in red
    for ligand_node, protein_node in zip(ligand_nodes, protein_nodes):
        if ligand_node < x_reduced.shape[0] and protein_node < y_reduced.shape[0]:  # Ensure indices are valid
            ax.plot([x_reduced[ligand_node, 0], y_reduced[protein_node, 0]],
                    [x_reduced[ligand_node, 1], y_reduced[protein_node, 1]],
                    [x_reduced[ligand_node, 2], y_reduced[protein_node, 2]],
                    c='red', alpha=0.8)  # Red edges

    # Add colorbar
    sm = ScalarMappable(cmap=cmap, norm=norm)  # Create ScalarMappable for colorbar
    sm.set_array([])  # Empty array for colorbar
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Distance (Euclidean)')

    # Set labels and title
    ax.set_xlabel('t-SNE Component 1')
    ax.set_ylabel('t-SNE Component 2')
    ax.set_zlabel('t-SNE Component 3')
    ax.set_title('3D t-SNE Visualization with Top 20% Closest Connections (Colored by Distance)')

    # Add legend
    ax.legend()

    # Save the plot
    plt.savefig(f'attention_map/tsne/{mode}_map.png', bbox_inches='tight')
    print(f"3D t-SNE plot with top 20% closest connections and edges from edge_index_dict saved.")


# def plot_tsne_with_attention(edge_index_dict, x_data, y_data, x_mask, y_mask, mode='lig', min_dist=0, max_dist=0.3):
#     print('t-SNE with distance-based connection, normalization, and edge coloring')

#     ligand_nodes = edge_index_dict[('ligand', 'to', 'protein')][0]
#     protein_nodes = edge_index_dict[('ligand', 'to', 'protein')][1]

#     num_ligand_nodes = x_data[x_mask].shape[0]
#     num_protein_nodes = y_data[y_mask].shape[0]
#     adj_matrix = np.zeros((num_ligand_nodes, num_protein_nodes))
#     for ligand_node, protein_node in zip(ligand_nodes, protein_nodes):
#         adj_matrix[ligand_node, protein_node] = 1  # 如果有边，设置为 0（深蓝色）

#     # Filter x_data and y_data using the provided masks
#     x_data = x_data[x_mask]  # Filter valid x nodes
#     y_data = y_data[y_mask]  # Filter valid y nodes

#     # Convert x_data and y_data to numpy arrays for t-SNE
#     x_data_np = x_data.detach().cpu().numpy()  # Convert to NumPy (for t-SNE)
#     y_data_np = y_data.detach().cpu().numpy()  # Convert to NumPy (for t-SNE)

#     # Combine x_data and y_data for t-SNE (for joint visualization)
#     combined_data = np.vstack([x_data_np, y_data_np])

#     # Apply t-SNE to reduce dimensionality to 3D
#     tsne = TSNE(n_components=3, random_state=42)
#     reduced_data = tsne.fit_transform(combined_data)

#     # Normalize the t-SNE reduced data using Min-Max normalization
#     scaler = MinMaxScaler()
#     reduced_data_normalized = scaler.fit_transform(reduced_data)  # Normalize the data to [0, 1]

#     # Split back the normalized reduced data into x_data and y_data components
#     x_reduced = reduced_data_normalized[:x_data_np.shape[0]]
#     y_reduced = reduced_data_normalized[x_data_np.shape[0]:]

#     # Calculate all distances between points in the 3D space
#     distances = []
#     for i in range(x_reduced.shape[0]):
#         for j in range(y_reduced.shape[0]):
#             distance = np.linalg.norm(x_reduced[i] - y_reduced[j])  # Euclidean distance in 3D space
#             distances.append((i, j, distance))

#     # Sort distances by the smallest distance and select top 20%
#     distances_sorted = sorted(distances, key=lambda x: x[2])  # Sort by distance (ascending)
#     top_20_percent_count = int(len(distances_sorted) * 0.01)  # Get the top 20%

#     # Plotting the histogram of distances
#     sorted_distances = [dist[2] for dist in distances_sorted]
#     plt.figure(figsize=(8, 6))
#     plt.hist(sorted_distances, bins=50, color='skyblue', edgecolor='black')
#     plt.xlim(0,1.25)  # Set x-axis limits
#     plt.ylim(0,300)  # Set y-axis limits
#     plt.title('Histogram of Sorted Distances Between Ligand and Protein Nodes')
#     plt.xlabel('Distance (Euclidean)')
#     plt.ylabel('Frequency')
#     plt.grid(True)
#     plt.savefig(f'attention_map/tsne/{mode}_distance_histogram.png', bbox_inches='tight')
#     print(f"Distance histogram saved.")

#     # Prepare color mapping for edges (using the distance as the color scale)
#     distance_values = [dist[2] for dist in distances_sorted[:top_20_percent_count]]  # Get distances of top 20%
    
#     # Use the provided min_dist and max_dist for the colorbar range
#     norm = Normalize(vmin=min_dist, vmax=max_dist)  # Normalize distances for colormap
#     cmap = plt.cm.viridis  # Choose a colormap

#     # Plotting
#     fig = plt.figure(figsize=(12, 10))
#     ax = fig.add_subplot(111, projection='3d')

#     # Plot x_data points (ligand nodes) in one color
#     ax.scatter(x_reduced[:, 0], x_reduced[:, 1], x_reduced[:, 2], c='blue', label='Ligand Nodes', alpha=0.6)

#     # Plot y_data points (protein nodes) in another color
#     ax.scatter(y_reduced[:, 0], y_reduced[:, 1], y_reduced[:, 2], c='red', label='Protein Nodes', alpha=0.6)

#     # Plot lines for the top 20% closest points, with colors based on distance
#     for i, j, distance in distances_sorted[:top_20_percent_count]:
#         if distance < 0.2:
#             color = cmap(norm(distance))  # Get color based on the distance
#             ax.plot([x_reduced[i, 0], y_reduced[j, 0]], 
#                     [x_reduced[i, 1], y_reduced[j, 1]], 
#                     [x_reduced[i, 2], y_reduced[j, 2]], 
#                     c=color, alpha=0.5)  # Line between closest points with color

#     # Add colorbar
#     sm = ScalarMappable(cmap=cmap, norm=norm)  # Create ScalarMappable for colorbar
#     sm.set_array([])  # Empty array for colorbar
#     cbar = plt.colorbar(sm, ax=ax)
#     cbar.set_label('Distance (Euclidean)')

#     # Set labels and title
#     ax.set_xlabel('t-SNE Component 1')
#     ax.set_ylabel('t-SNE Component 2')
#     ax.set_zlabel('t-SNE Component 3')
#     ax.set_title('3D t-SNE Visualization with Top 20% Closest Connections (Colored by Distance)')

#     # Add legend
#     ax.legend()

#     # Save the plot
#     plt.savefig(f'attention_map/tsne/{mode}_map.png', bbox_inches='tight')
#     print(f"3D t-SNE plot with top 20% closest connections colored by distance saved.")



class HeteroGNN(torch.nn.Module):
    def __init__(self, metadata, edge_dim, hidden_channels, out_channels, num_layers, mode):
        super().__init__()
        # print(metadata)
        self.mode = mode
        self.edge_mlps = torch.nn.ModuleDict({
            str(edge_type): MLP([hidden_channels * 2 + out_channels, 512, 64, 16], dropout=0.1)#str(edge_type): MLP([hidden_channels * 2 + 8, 512, 64, 16], dropout=0.1)
            for edge_type in metadata[1]
        })
        self.edge_lins = torch.nn.ModuleDict({
            str(edge_type): Linear(in_channels=1, out_channels=8)
            for edge_type in metadata[1]
        })
        self.lin_mpl = Linear(in_channels=16, out_channels=16)#out_channels=16

        metadata_v2pl = [('env', 'to', 'protein'), ('env', 'to', 'ligand')]
        metadata_pl2v = [('protein', 'rev_to', 'env'), ('ligand', 'rev_to', 'env')]
        metadata_pl = [('ligand', 'to', 'protein'), ('protein', 'rev_to', 'ligand'), ('protein', 'to', 'protein')]
        if('noEnv' in self.mode):
            self.edge_index_pl = [('ligand', 'to', 'protein')]
        else:
            self.edge_index_pl = [('env', 'to', 'protein'), ('env', 'to', 'ligand'), ('ligand', 'to', 'protein')]

        # virtual to protein/ligand
        self.convs_v2pl = torch.nn.ModuleList([HeteroConv({
            edge_type: SAGEConv((-1, -1), hidden_channels)
            for edge_type in metadata_v2pl
        }) for _ in range(num_layers)])

        # protein/ligand to virtual
        self.convs_pl2v = torch.nn.ModuleList([HeteroConv({
            edge_type: SAGEConv((-1, -1), hidden_channels)
            for edge_type in metadata_pl2v
        }) for _ in range(num_layers)])

        # protein/ligand to protein/ligand
        self.convs_pl = torch.nn.ModuleList([HeteroConv({
            edge_type: SAGEConv((-1, -1), hidden_channels)
            for edge_type in metadata_pl
        }) for _ in range(num_layers)])

    def forward(self, x_dict, edge_index_dict, edge_attr_dict, batch_dict):
        ligand_sum = 0
        protein_sum = 0
        virtual_sum = 0
        # print(x_dict['env'].shape)
        # for i in range(0, x_dict['env'].shape[0]):
        #     if(np.mod(i,3)==0):
        #         # print('置零')
        #         x_dict['env'][i+2][3]=0
        for i in range(len(self.convs_pl)):
            if('validation' in self.mode and i==0):
                lig, lig_mask = to_dense_batch(x_dict['ligand'], batch_dict['ligand'], max_num_nodes=1024)
                pro, pro_mask = to_dense_batch(x_dict['protein'], batch_dict['protein'], max_num_nodes=1024)
                env, env_mask = to_dense_batch(x_dict['env'], batch_dict['env'], max_num_nodes=3)
                for j in range(0, lig.shape[0]):
                    global n2
                    if('noEnv' in self.mode):
                        md = f'noEnv_{n2}_{i}'
                    elif('noHGT' in self.mode):
                        md = f'noHGT_{n2}_{i}'
                    elif('oHGT' in self.mode):
                        md = f'oHGT_{n2}_{i}'
                    else:
                        md = f'full_{n2}_{i}'
                    fig = plot_tsne_with_attention(edge_index_dict, lig[j], pro[j], lig_mask[j], pro_mask[j], mode=md)
                    fig = plot_weisfeiler_lehman(lig[j], pro[j], lig_mask[j], pro_mask[j], mode=md, edge_index_dict=edge_index_dict, edge_attr_dict=edge_attr_dict)
                    # plot_edge(edge_index_dict, lig[j][lig_mask[j]].shape[0], pro[j][pro_mask[j]].shape[0])
                    n2+=1
                    print(n2)

            if('noEnv' in self.mode):
                pl_dict = self.convs_pl[i](x_dict, edge_index_dict)
                pl_dict = {key: F.leaky_relu(x) for key, x in pl_dict.items()}
                x_dict.update(pl_dict)
            else:
                v2pl_dict = self.convs_v2pl[i](x_dict, edge_index_dict)
                v2pl_dict = {key: F.leaky_relu(x) for key, x in v2pl_dict.items()}
                x_dict.update(v2pl_dict)

                pl_dict = self.convs_pl[i](x_dict, edge_index_dict)
                # print(pl_dict.items())
                pl_dict = {key: F.leaky_relu(x) for key, x in pl_dict.items()}
                x_dict.update(pl_dict)

                pl2v_dict = self.convs_pl2v[i](x_dict, edge_index_dict)
                pl2v_dict = {key: F.leaky_relu(x) for key, x in pl2v_dict.items()}
                x_dict.update(pl2v_dict)
            
            ligand_sum += x_dict['ligand']
            protein_sum += x_dict['protein']
            virtual_sum += x_dict['env']

            if('validation' in self.mode):
                lig, lig_mask = to_dense_batch(ligand_sum, batch_dict['ligand'], max_num_nodes=1024)
                pro, pro_mask = to_dense_batch(protein_sum, batch_dict['protein'], max_num_nodes=1024)
                env, env_mask = to_dense_batch(virtual_sum, batch_dict['env'], max_num_nodes=3)
                for j in range(0, lig.shape[0]):
                    global n
                    if('noEnv' in self.mode):
                        md = f'noEnv_{n}_{i+1}'
                    else:
                        md = f'full_{n}_{i+1}'
                    fig = plot_tsne_with_attention(edge_index_dict, lig[j], pro[j], lig_mask[j], pro_mask[j], mode=md)
                    fig = plot_weisfeiler_lehman(lig[j], pro[j], lig_mask[j], pro_mask[j], mode=md, edge_index_dict=edge_index_dict, edge_attr_dict=edge_attr_dict)
                    # plot_edge(edge_index_dict, lig[j][lig_mask[j]].shape[0], pro[j][pro_mask[j]].shape[0])
                    n+=1
                    print(n)

        x_dict['ligand'] = ligand_sum
        x_dict['protein'] = protein_sum
        x_dict['env'] = virtual_sum

        edge_outputs = []
        for edge_type in self.edge_index_pl:
            # print(edge_type)
            # print(edge_attr_dict.keys())
            src, dst = edge_index_dict[edge_type]
            src_type, _, dst_type = edge_type
            edge_repr = torch.cat([x_dict[src_type][src], x_dict[dst_type][dst]], dim=-1)
            d_edge = self.edge_lins[str(edge_type)](edge_attr_dict[edge_type])

            edge_repr = torch.cat((edge_repr, d_edge), dim=1)

            m_edge = self.edge_mlps[str(edge_type)](edge_repr)

            edge_batch = batch_dict[src_type][src]
            w_edge = torch.tanh(self.lin_mpl(m_edge))
            m_w = w_edge * m_edge

            m_w = scatter_sum(m_w, edge_batch, dim=0)
            m_max, _ = scatter_max(m_edge, edge_batch, dim=0)
            edge_outputs.append(torch.cat((m_w, m_max), dim=1))

        m_out = torch.cat(edge_outputs, dim=1)
        return m_out