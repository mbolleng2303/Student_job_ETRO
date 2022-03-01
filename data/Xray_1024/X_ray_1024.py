import numpy as np, time, pickle, random, csv
import torch
from torch.utils.data import DataLoader, Dataset

import os
import pickle
import numpy as np
import torch
import dgl

from sklearn.model_selection import StratifiedKFold, train_test_split

random.seed(42)

from scipy import sparse as sp


class DGLFormDataset(torch.utils.data.Dataset):
    """
        DGLFormDataset wrapping graph list and label list as per pytorch Dataset.
        *lists (list): lists of 'graphs' and 'labels' with same len().
    """

    def __init__(self, *lists):
        assert all(len(lists[0]) == len(li) for li in lists)
        self.lists = lists
        self.graph_lists = lists[0]
        self.label_lists = lists[1]

    def __getitem__(self, index):
        return tuple(li[index] for li in self.lists)

    def __len__(self):
        return len(self.lists[0])


def format_dataset(dataset):
    """
        Utility function to recover data,
        INTO-> dgl/pytorch compatible format
    """
    graphs = [data[0] for data in dataset]
    labels = [data[1] for data in dataset]

    for graph in graphs:
        # graph.ndata['feat'] = torch.FloatTensor(graph.ndata['feat'])
        graph.ndata['feat'] = graph.ndata['feat'].float()  # dgl 4.0
        # adding edge features for Residual Gated ConvNet, if not there
        if 'feat' not in graph.edata.keys():
            edge_feat_dim = graph.ndata['feat'].shape[1]  # dim same as node feature dim
            graph.edata['feat'] = torch.ones(graph.number_of_edges(), edge_feat_dim)

    return DGLFormDataset(graphs, labels)


def get_all_split_idx(dataset):
    """
        - Split total number of graphs into 3 (train, val and test) in 3:1:1
        - Stratified split proportionate to original distribution of data with respect to classes
        - Using sklearn to perform the split and then save the indexes
        - Preparing 5 such combinations of indexes split to be used in Graph NNs
        - As with KFold, each of the 5 fold have unique test set.
    """
    root_idx_dir = "C:/Users/Surface/PycharmProjects/Student_job/data/Xray_1024/"
    if not os.path.exists(root_idx_dir):
        os.makedirs(root_idx_dir)
    all_idx = {}

    # If there are no idx files, do the split and store the files
    if not (os.path.exists(root_idx_dir + dataset.name + '_train.csv')):
        print("[!] Splitting the data into train/val/test ...")

        # Using 5-fold cross val as used in RP-GNN paper
        k_splits = 5

        cross_val_fold = StratifiedKFold(n_splits=k_splits, shuffle=True)
        k_data_splits = []

        # this is a temporary index assignment, to be used below for val splitting
        for i in range(len(dataset.graph_lists)):
            dataset[i][0].a = lambda: None
            setattr(dataset[i][0].a, 'index', i)
        a=np.random.rand(len(dataset.graph_lists))
        for i in range (len(a)) :
            a[i]= 1
        f_train_w = []
        f_val_w = []
        f_test_w = []
        for indexes in cross_val_fold.split(dataset.graph_lists, a):
            remain_index, test_index = indexes[0], indexes[1]

            remain_set = format_dataset([dataset[index] for index in remain_index])

            # Gets final 'train' and 'val'
            train, val, _, __ = train_test_split(remain_set,
                                                 range(len(remain_set.graph_lists)),
                                                 test_size=0.25)
            train, val = format_dataset(train), format_dataset(val)
            test = format_dataset([dataset[index] for index in test_index])

            # Extracting only idxs
            idx_train = [item[0].a.index for item in train]
            idx_val = [item[0].a.index for item in val]
            idx_test = [item[0].a.index for item in test]
            """
            f_train_w = csv.writer(open(root_idx_dir + dataset.name + '_train.csv', 'a+'))
            f_val_w = csv.writer(open(root_idx_dir + dataset.name + '_val.csv', 'a+'))
            f_test_w = csv.writer(open(root_idx_dir + dataset.name + '_test.csv', 'a+'))
""          """
            f_train_w.append(idx_train)
            f_val_w.append(idx_val)
            f_test_w.append(idx_test)


        # reading idx from the files
        f_train_w = np.array(f_train_w)
        f_val_w = np.array(f_val_w)
        f_test_w = np.array(f_test_w)
        f_train_w.tofile(root_idx_dir +'X_ray_1024_train.csv', sep=',')
        f_val_w.tofile(root_idx_dir +'X_ray_1024_val.csv', sep=',')
        f_test_w.tofile(root_idx_dir +'X_ray_1024_test.csv', sep=',')
        print("[!] Splitting done!")


    for section in ['train', 'val', 'test']:
        all_idx[section] = np.reshape(np.loadtxt(root_idx_dir +"X_ray_1024_"+ section +".csv",
                delimiter=",", dtype=int), (k_splits,-1 ))
    return all_idx

def get_vertices(a):
    edges = []
    feat = []
    for i in range(a.shape[1]):
        for j in range(i, a.shape[0]):
            if a[i, j] != 0:
                edges.append((i, j))
                feat.append(a[i, j])
                # edges.append((j, i)) #for two dir
    return edges, feat
class Data2Graph(torch.utils.data.Dataset):
    """
        Circular Skip Link Graphs:
        Source: https://github.com/PurdueMINDS/RelationalPooling/
    """

    def __init__(self, path="data/Xray_1024/"):
        self.nbr_graphs = 100
        self.nbr_node = 10
        self.name = "Xray_1024"
        self.graph= np.reshape(np.loadtxt('C:/Users/Surface/PycharmProjects/Student_job/data/Xray_1024/X_ray_1024_node.csv',
                                delimiter=",", dtype=float), (1024, self.nbr_graphs, self.nbr_node))
        self.edge = np.reshape(np.loadtxt('C:/Users/Surface/PycharmProjects/Student_job/data/Xray_1024/X_ray_1024_edge.csv',
                 delimiter=",", dtype=float), (self.nbr_graphs,self.nbr_node,self.nbr_node))

        self.label = np.reshape(np.loadtxt('C:/Users/Surface/PycharmProjects/Student_job/data/Xray_1024/X_ray_1024_label.csv',
                 delimiter=",", dtype=float), (1,self.nbr_graphs,self.nbr_node))

        self.graph_lists = []
        self.label_lists = []


        #self.n_samples = len(selflabel_lists)
        #self.num_node_type = 1  # 41
        #self.num_edge_type = 1  # 164
        self._prepare()

    def _prepare(self):
        t0 = time.time()
        print("[I] Preparing Circular Skip Link Graphs v4 ...")
        for i in range (self.nbr_graphs):
            g = dgl.DGLGraph()


            #label = dgl.DGLGraph()


            g.add_nodes(self.nbr_node)
            #label.add_nodes(self.nbr_node )
            # S3: add edges using g.add_edges()
            g.ndata['feat'] = torch.tensor(self.graph[:,i,:].T).long()
            #label.ndata['feat'] = torch.tensor(self.label[:, i, :].T).long()
            # g.ndata['feat'] = torch.arange(0, g.number_of_nodes()).long() # v1
            # g.ndata['feat'] = torch.randperm(g.number_of_nodes()).long() # v3

            # adding edge features as generic requirement
            edge = np.array(get_vertices(self.edge[i,:,:])[0])
            edge_feat = np.array(get_vertices(self.edge[i,:,:])[1])
            for src, dst in edge:
                g.add_edges(src.item(), dst.item())
                #label.add_edges(src.item(), dst.item())
                #edge_feat.append(self.edge[i,src.item(),dst.item()])
            edge_feat_dim = 1
            edge_feat=np.array(edge_feat)
            g.edata['feat'] = torch.tensor((edge_feat)).long()
            #label.edata['feat'] = torch.tensor((edge_feat)).long()
            # g.edata['feat'] = torch.arange(0, g.number_of_edges()).long() # v1
            # g.edata['feat'] = torch.ones(g.number_of_edges()).long() # v2
            g = dgl.transform.remove_self_loop(g)
            # NOTE: come back here, to define edge features as distance between the indices of the edges
            ###################################################################
            # srcs, dsts = new_g.edges()
            # edge_feat = []
            # for edge in range(len(srcs)):
            #     a = srcs[edge].item()
            #     b = dsts[edge].item()
            #     edge_feat.append(abs(a-b))
            # g.edata['feat'] = torch.tensor(edge_feat, dtype=torch.int).long()
            ############################
            # #######################################
            a = (self.label[0, i, :])
            res = [0,0,0,0]
            b=[]
            for i in a :
                res[int(i)]=1
                b.append(res)
                res = [0,0,0,0]
            self.label_lists.append(torch.tensor(b))
            self.graph_lists.append(g)
        #self.num_node_type = self.graph_lists[0].ndata['feat'].size(0)
        #self.num_edge_type = self.graph_lists[0].edata['feat'].size(0)
        print("[I] Finished preparation after {:.4f}s".format(time.time() - t0))

    def __len__(self):
        return self.nbr_graphs

    def __getitem__(self, idx):
        return self.graph_lists[idx], self.label_lists[idx]


def self_loop(g):
    """
        Utility function only, to be used only when necessary as per user self_loop flag
        : Overwriting the function dgl.transform.add_self_loop() to not miss ndata['feat'] and edata['feat']


        This function is called inside a function in TUsDataset class.
    """
    new_g = dgl.DGLGraph()
    new_g.add_nodes(g.number_of_nodes())
    new_g.ndata['feat'] = g.ndata['feat']

    src, dst = g.all_edges(order="eid")
    src = dgl.backend.zerocopy_to_numpy(src)
    dst = dgl.backend.zerocopy_to_numpy(dst)
    non_self_edges_idx = src != dst
    nodes = np.arange(g.number_of_nodes())
    new_g.add_edges(src[non_self_edges_idx], dst[non_self_edges_idx])
    new_g.add_edges(nodes, nodes)

    # This new edata is not used since this function gets called only for GCN, GAT
    # However, we need this for the generic requirement of ndata and edata
    new_g.edata['feat'] = torch.zeros(new_g.number_of_edges())
    return new_g


def self_loop(g):
    """
        Utility function only, to be used only when necessary as per user self_loop flag
        : Overwriting the function dgl.transform.add_self_loop() to not miss ndata['feat'] and edata['feat']


        This function is called inside a function in TUsDataset class.
    """
    new_g = dgl.DGLGraph()
    new_g.add_nodes(g.number_of_nodes())
    new_g.ndata['feat'] = g.ndata['feat']

    src, dst = g.all_edges(order="eid")
    src = dgl.backend.zerocopy_to_numpy(src)
    dst = dgl.backend.zerocopy_to_numpy(dst)
    non_self_edges_idx = src != dst
    nodes = np.arange(g.number_of_nodes())
    new_g.add_edges(src[non_self_edges_idx], dst[non_self_edges_idx])
    new_g.add_edges(nodes, nodes)

    # This new edata is not used since this function gets called only for GCN, GAT
    # However, we need this for the generic requirement of ndata and edata
    new_g.edata['feat'] = torch.zeros(new_g.number_of_edges())
    return new_g

def positional_encoding(g, pos_enc_dim):
    """
        Graph positional encoding v/ Laplacian eigenvectors
    """
    n = g.number_of_nodes()
    # Laplacian
    A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
    N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
    L = sp.eye(n) - N * A * N
    # Eigenvectors
    EigVal, EigVec = np.linalg.eig(L.toarray())
    idx = EigVal.argsort()  # increasing order
    EigVal, EigVec = EigVal[idx], np.real(EigVec[:, idx])
    g.ndata['pos_enc'] = torch.from_numpy(EigVec[:, 1:pos_enc_dim + 1]).float()
    return g


"collate"
class X_ray_1024_Dataset(torch.utils.data.Dataset):
    def __init__(self, name='X_ray_1024'):
        t0 = time.time()
        self.name = name

        dataset = Data2Graph()

        print("[!] Dataset: ", self.name)

        # this function splits data into train/val/test and returns the indices
        self.all_idx = get_all_split_idx(dataset)

        self.all = dataset
        self.train = [self.format_dataset([dataset[idx] for idx in self.all_idx['train'][split_num]]) for split_num in
                      range(5)]
        self.val = [self.format_dataset([dataset[idx] for idx in self.all_idx['val'][split_num]]) for split_num in
                    range(5)]
        self.test = [self.format_dataset([dataset[idx] for idx in self.all_idx['test'][split_num]]) for split_num in
                     range(5)]

        print("Time taken: {:.4f}s".format(time.time() - t0))

    def format_dataset(self, dataset):
        """
            Utility function to recover data,
            INTO-> dgl/pytorch compatible format
        """
        graphs = [data[0] for data in dataset]
        labels = [data[1] for data in dataset]

        return DGLFormDataset(graphs, labels)

    # form a mini batch from a given list of samples = [(graph, label) pairs]
    def collate(self, samples):
        # The input samples is a list of pairs (graph, label).
        graphs, labels = map(list, zip(*samples))
        labels = torch.cat(labels).long()
        batched_graph = dgl.batch(graphs)
        return batched_graph, labels

    # prepare dense tensors for GNNs using them; such as RingGNN, 3WLGNN
    def collate_dense_gnn(self, samples, pos_enc):
        # The input samples is a list of pairs (graph, label).
        graphs, labels = map(list, zip(*samples))
        labels = torch.tensor(np.array(labels))
        g = graphs[0]
        adj = self._sym_normalize_adj(g.adjacency_matrix().to_dense())
        """
            Adapted from https://github.com/leichen2018/Ring-GNN/
            Assigning node and edge feats::
            we have the adjacency matrix in R^{n x n}, the node features in R^{d_n} and edge features R^{d_e}.
            Then we build a zero-initialized tensor, say T, in R^{(1 + d_n + d_e) x n x n}. T[0, :, :] is the adjacency matrix.
            The diagonal T[1:1+d_n, i, i], i = 0 to n-1, store the node feature of node i. 
            The off diagonal T[1+d_n:, i, j] store edge features of edge(i, j).
        """
        zero_adj = torch.zeros_like(adj)
        if pos_enc:
            in_dim = g.ndata['pos_enc'].shape[1]
            # use node feats to prepare adj
            adj_node_feat = torch.stack([zero_adj for j in range(in_dim)])
            adj_node_feat = torch.cat([adj.unsqueeze(0), adj_node_feat], dim=0)
            for node, node_feat in enumerate(g.ndata['pos_enc']):
                adj_node_feat[1:, node, node] = node_feat
            x_node_feat = adj_node_feat.unsqueeze(0)
            return x_node_feat, labels
        else:  # no node features here
            in_dim = 1
            # use node feats to prepare adj
            adj_node_feat = torch.stack([zero_adj for j in range(in_dim)])
            adj_node_feat = torch.cat([adj.unsqueeze(0), adj_node_feat], dim=0)
            for node, node_feat in enumerate(g.ndata['feat']):
                adj_node_feat[1:, node, node] = node_feat
            x_no_node_feat = adj_node_feat.unsqueeze(0)
            return x_no_node_feat, labels

    def _sym_normalize_adj(self, adj):
        deg = torch.sum(adj, dim=0)  # .squeeze()
        deg_inv = torch.where(deg > 0, 1. / torch.sqrt(deg), torch.zeros(deg.size()))
        deg_inv = torch.diag(deg_inv)
        return torch.mm(deg_inv, torch.mm(adj, deg_inv))

    def _add_self_loops(self):

        # function for adding self loops
        # this function will be called only if self_loop flag is True
        for split_num in range(5):
            self.train[split_num].graph_lists = [self_loop(g) for g in self.train[split_num].graph_lists]
            self.val[split_num].graph_lists = [self_loop(g) for g in self.val[split_num].graph_lists]
            self.test[split_num].graph_lists = [self_loop(g) for g in self.test[split_num].graph_lists]

        for split_num in range(5):
            self.train[split_num] = DGLFormDataset(self.train[split_num].graph_lists,
                                                   self.train[split_num].label_lists)
            self.val[split_num] = DGLFormDataset(self.val[split_num].graph_lists, self.val[split_num].label_lists)
            self.test[split_num] = DGLFormDataset(self.test[split_num].graph_lists, self.test[split_num].label_lists)

    def _add_positional_encodings(self, pos_enc_dim):

        # Graph positional encoding v/ Laplacian eigenvectors
        for split_num in range(5):
            self.train[split_num].graph_lists = [positional_encoding(g, pos_enc_dim) for g in
                                                 self.train[split_num].graph_lists]
            self.val[split_num].graph_lists = [positional_encoding(g, pos_enc_dim) for g in
                                               self.val[split_num].graph_lists]
            self.test[split_num].graph_lists = [positional_encoding(g, pos_enc_dim) for g in
                                                self.test[split_num].graph_lists]