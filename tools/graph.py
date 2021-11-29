import numpy as np


def create_adj_matrix(adj_vector, networksize):
    adj_matrix = np.zeros((networksize, networksize))

    adj_matrix[np.triu_indices(networksize, k=1)] = adj_vector[
        0 : np.shape(np.triu_indices(networksize, k=1))[1]
    ]
    adj_matrix = adj_matrix + np.transpose(adj_matrix)

    return adj_matrix


def create_position_matrix(position_vector, cut_off_size=None):
    length = int(len(position_vector) / 2)
    position_matrix = position_vector.reshape(length, 2)
    if cut_off_size == None:
        cut_off_size = length
    position_matrix = position_matrix[0:cut_off_size, :]
    return position_matrix


def create_graph_mask(network_dim, pos: np.array, adjacency: np.ndarray):
    node_adj_mask = np.zeros((network_dim, network_dim))
    positions = pos.astype(int)
    for x_pixel in range(len(adjacency)):
        for y_pixel in range(len(adjacency)):
            node_adj_mask[positions[x_pixel][0], positions[y_pixel][1]] = adjacency[
                x_pixel, y_pixel
            ]

    return node_adj_mask


def create_graph_tensor(mask):
    """Mask Normalization
    Function that returns normalized mask
    Each pixel is either 0 or 1
    """
    mask = np.asarray(mask)
    y_positions_label = mask[:, 0:2, 0]
    y_adjacency_label = mask[:, 2:, 0]
    pos = y_positions_label.astype(int)
    adj_dim = int((len(pos) * len(pos) - len(pos)) / 2)
    tensor_graph = np.zeros(adj_dim)

    for node_idx in range(len(pos)):
        adjacency_idx_vec = np.argwhere(
            y_adjacency_label[node_idx, :] == 1
        )  # find all set of nodes that are connected to the one respective node_idx (this is evaluated for all rows)
        idx_trivec = get_indices_trivec_adjacency(len(pos), node_idx, adjacency_idx_vec)
        tensor_graph[idx_trivec] = 1  # label
    tensor_graph.astype(int)

    return tensor_graph


def create_graph_vec_fixed_dim(adj, dim_nr_nodes=128):
    """Mask Normalization
    Function that returns normalized mask
    Each pixel is either 0 or 1
    """
    adj_dim = int((dim_nr_nodes * dim_nr_nodes - dim_nr_nodes) / 2)
    tensor_graph = np.zeros(adj_dim)
    for node_idx in range(len(adj[:, 0])):
        adjacency_idx_vec = np.argwhere(
            adj[node_idx, :] == 1
        )  # find all set of nodes that are connected to the one respective node_idx (this is evaluated for all rows)
        # global_adj_idx_tuple = pos[adjacency_idx, :]  # position of the connected node
        idx_trivec = get_indices_trivec_adjacency(
            dim_nr_nodes, node_idx, adjacency_idx_vec
        )
        # tensor_graph[node_idx[0], node_idx[1], idx_trivec] = 1 #label
        tensor_graph[idx_trivec] = 1  # label
    tensor_graph.astype(int)

    return tensor_graph


def get_indices_trivec_adjacency(node_dim, node_idx, adjacency_idx_vec):
    adj_tri_vec_idx = np.triu_indices(node_dim, k=1)
    tmp_matrix = np.full((node_dim, node_dim), False, dtype=bool)
    tmp_matrix[node_idx, adjacency_idx_vec] = True
    adj_tri_vec = tmp_matrix[adj_tri_vec_idx]
    idx_trivec = np.argwhere(adj_tri_vec == True)
    idx_trivec[np.lexsort(np.fliplr(idx_trivec).T)]

    # idx_trivec = np.sort(idx_trivec, axis=0) # ensure that idx_trivec is correclty sorted
    return np.squeeze(idx_trivec)


def tensor_2_adjmatrix(adj_vector, networksize, nr_nodes):
    adj_matrix = np.zeros((networksize, networksize))
    # adj_matrix[np.triu_indices(networksize, k = 1)] = adj_vector[0:np.shape(np.triu_indices(networksize, k = 1))[1]]
    adj_matrix[np.triu_indices(networksize, k=1)] = adj_vector
    adj_matrix = adj_matrix + np.transpose(adj_matrix)
    # adj_matrix = W = np.maximum( adj_matrix, adj_matrix.T)
    adj_matrix = adj_matrix[:nr_nodes, :nr_nodes]
    adj_matrix_tmp = adj_matrix.copy()
    adj_matrix_tmp[:nr_nodes, :nr_nodes] = 0
    print("Is just True iff all entries are zero ::", np.all(adj_matrix_tmp == 0))

    return adj_matrix


def create_input_image_node_tensor(img, nodes, size):
    """Returns a bilayer image:
    On first layer: original image
    On second layer: node positions
    """
    tensor = np.zeros((size[0], size[1], 2))
    tensor[:, :, 0] = img
    nodes = nodes.astype(int)
    for index in range(len(nodes)):
        tensor[nodes[index, 0], nodes[index, 1], 1] = 1
    return tensor


def tensor_2_image_and_pos(tensor):
    image = tensor[:, :, 0] * 255
    image = np.array(image, dtype="float32")
    pos_matrix = tensor[:, :, 1].astype(int)
    pos = np.argwhere(pos_matrix > 0)
    # pos = np.sort(pos)
    # print('output pos after where',pos)
    pos = np.asarray(pos)
    pos[np.lexsort(np.fliplr(pos).T)]
    # pos = np.sort(pos, axis=0)   # new to ensure correct order, this is maybe redundantp
    print("len of pos", len(pos))
    # print('shape of postion',pos.shape)

    return image, pos
