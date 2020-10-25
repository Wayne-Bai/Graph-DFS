import torch
import torchvision as tv
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
from random import shuffle

import networkx as nx
import pickle as pkl
import scipy.sparse as sp
import logging

import random
import shutil
import os
import time
from model import *
from utils import *

import create_graphs



# load ENZYMES and PROTEIN and DD dataset
def Graph_load_batch(min_num_nodes = 20, max_num_nodes = 1000, name = 'ENZYMES',node_attributes = True,graph_labels=True):
    '''
    load many graphs, e.g. enzymes
    :return: a list of graphs
    '''
    print('Loading graph dataset: '+str(name))
    G = nx.Graph()
    # load data
    path = 'dataset/'+name+'/'
    data_adj = np.loadtxt(path+name+'_A.txt', delimiter=',').astype(int)
    if node_attributes:
        data_node_att = np.loadtxt(path+name+'_node_attributes.txt', delimiter=',')
    data_node_label = np.loadtxt(path+name+'_node_labels.txt', delimiter=',').astype(int)
    data_graph_indicator = np.loadtxt(path+name+'_graph_indicator.txt', delimiter=',').astype(int)
    if graph_labels:
        data_graph_labels = np.loadtxt(path+name+'_graph_labels.txt', delimiter=',').astype(int)


    data_tuple = list(map(tuple, data_adj))
    # print(len(data_tuple))
    # print(data_tuple[0])

    # add edges
    G.add_edges_from(data_tuple)
    # add node attributes
    for i in range(data_node_label.shape[0]):
        if node_attributes:
            G.add_node(i+1, feature = data_node_att[i])
        G.add_node(i+1, label = data_node_label[i])
    G.remove_nodes_from(list(nx.isolates(G)))

    # print(G.number_of_nodes())
    # print(G.number_of_edges())

    # split into graphs
    graph_num = data_graph_indicator.max()
    node_list = np.arange(data_graph_indicator.shape[0])+1
    graphs = []
    max_nodes = 0
    for i in range(graph_num):
        # find the nodes for each graph
        nodes = node_list[data_graph_indicator==i+1]
        G_sub = G.subgraph(nodes)
        if graph_labels:
            G_sub.graph['label'] = data_graph_labels[i]
        # print('nodes', G_sub.number_of_nodes())
        # print('edges', G_sub.number_of_edges())
        # print('label', G_sub.graph)
        if G_sub.number_of_nodes()>=min_num_nodes and G_sub.number_of_nodes()<=max_num_nodes:
            graphs.append(G_sub)
            if G_sub.number_of_nodes() > max_nodes:
                max_nodes = G_sub.number_of_nodes()
            # print(G_sub.number_of_nodes(), 'i', i)
    # print('Graph dataset name: {}, total graph num: {}'.format(name, len(graphs)))
    # logging.warning('Graphs loaded, total num: {}'.format(len(graphs)))
    print('Loaded')
    return graphs

def test_graph_load_DD():
    graphs, max_num_nodes = Graph_load_batch(min_num_nodes=10,name='DD',node_attributes=False,graph_labels=True)
    shuffle(graphs)
    plt.switch_backend('agg')
    plt.hist([len(graphs[i]) for i in range(len(graphs))], bins=100)
    plt.savefig('figures/test.png')
    plt.close()
    row = 4
    col = 4
    draw_graph_list(graphs[0:row*col], row=row,col=col, fname='figures/test')
    print('max num nodes',max_num_nodes)


def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

# load cora, citeseer and pubmed dataset
def Graph_load(dataset = 'cora'):
    '''
    Load a single graph dataset
    :param dataset: dataset name
    :return:
    '''
    names = ['x', 'tx', 'allx', 'graph']
    objects = []
    for i in range(len(names)):
        load = pkl.load(open("dataset/ind.{}.{}".format(dataset, names[i]), 'rb'), encoding='latin1')
        # print('loaded')
        objects.append(load)
        # print(load)
    x, tx, allx, graph = tuple(objects)
    test_idx_reorder = parse_index_file("dataset/ind.{}.test.index".format(dataset))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    G = nx.from_dict_of_lists(graph)
    adj = nx.adjacency_matrix(G)
    return adj, features, G


######### code test ########
# adj, features,G = Graph_load()
# print(adj)
# print(G.number_of_nodes(), G.number_of_edges())

# _,_,G = Graph_load(dataset='citeseer')
# G = max(nx.connected_component_subgraphs(G), key=len)
# G = nx.convert_node_labels_to_integers(G)
#
# count = 0
# max_node = 0
# for i in range(G.number_of_nodes()):
#     G_ego = nx.ego_graph(G, i, radius=3)
#     # draw_graph(G_ego,prefix='test'+str(i))
#     m = G_ego.number_of_nodes()
#     if m>max_node:
#         max_node = m
#     if m>=50:
#         print(i, G_ego.number_of_nodes(), G_ego.number_of_edges())
#         count += 1
# print('count', count)
# print('max_node', max_node)




def bfs_seq(G, start_id):
    '''
    get a bfs node sequence
    :param G:
    :param start_id:
    :return:
    '''
    dictionary = dict(nx.bfs_successors(G, start_id))
    start = [start_id]
    output = [start_id]
    while len(start) > 0:
        next = []
        while len(start) > 0:
            current = start.pop(0)
            neighbor = dictionary.get(current)
            if neighbor is not None:
                #### a wrong example, should not permute here!
                # shuffle(neighbor)
                next = next + neighbor
        output = output + next
        start = next
    return output

def dfs_seq(G, start_id):
    DFS = nx.dfs_tree(G, source=start_id)
    output = list(DFS)
    return output


def encode_adj(adj, max_prev_node=10, is_full = False):
    '''

    :param adj: n*n, rows means time step, while columns are input dimension
    :param max_degree: we want to keep row number, but truncate column numbers
    :return:
    '''
    if is_full:
        max_prev_node = adj.shape[0]-1

    # pick up lower tri
    adj = np.tril(adj, k=-1)
    n = adj.shape[0]
    adj = adj[1:n, 0:n-1]

    # use max_prev_node to truncate
    # note: now adj is a (n-1)*(n-1) matrix
    adj_output = np.zeros((adj.shape[0], max_prev_node))
    for i in range(adj.shape[0]):
        input_start = max(0, i - max_prev_node + 1)
        input_end = i + 1
        output_start = max_prev_node + input_start - input_end
        output_end = max_prev_node
        adj_output[i, output_start:output_end] = adj[i, input_start:input_end]
        adj_output[i,:] = adj_output[i,:][::-1] # reverse order

    return adj_output

def decode_adj(adj_output):
    '''
        recover to adj from adj_output
        note: here adj_output have shape (n-1)*m
    '''
    max_prev_node = adj_output.shape[1]
    adj = np.zeros((adj_output.shape[0], adj_output.shape[0]))
    for i in range(adj_output.shape[0]):
        input_start = max(0, i - max_prev_node + 1)
        input_end = i + 1
        output_start = max_prev_node + max(0, i - max_prev_node + 1) - (i + 1)
        output_end = max_prev_node
        adj[i, input_start:input_end] = adj_output[i,::-1][output_start:output_end] # reverse order
    adj_full = np.zeros((adj_output.shape[0]+1, adj_output.shape[0]+1))
    n = adj_full.shape[0]
    adj_full[1:n, 0:n-1] = np.tril(adj, 0)
    adj_full = adj_full + adj_full.T

    return adj_full


def encode_adj_flexible(adj):
    '''
    return a flexible length of output
    note that here there is no loss when encoding/decoding an adj matrix
    :param adj: adj matrix
    :return:
    '''
    # pick up lower tri
    adj = np.tril(adj, k=-1)
    n = adj.shape[0]
    adj = adj[1:n, 0:n-1]

    adj_output = []
    input_start = 0
    for i in range(adj.shape[0]):
        input_end = i + 1
        adj_slice = adj[i, input_start:input_end]
        adj_output.append(adj_slice)
        non_zero = np.nonzero(adj_slice)[0]
        input_start = input_end-len(adj_slice)+np.amin(non_zero)

    return adj_output



def decode_adj_flexible(adj_output):
    '''
    return a flexible length of output
    note that here there is no loss when encoding/decoding an adj matrix
    :param adj: adj matrix
    :return:
    '''
    adj = np.zeros((len(adj_output), len(adj_output)))
    for i in range(len(adj_output)):
        output_start = i+1-len(adj_output[i])
        output_end = i+1
        adj[i, output_start:output_end] = adj_output[i]
    adj_full = np.zeros((len(adj_output)+1, len(adj_output)+1))
    n = adj_full.shape[0]
    adj_full[1:n, 0:n-1] = np.tril(adj, 0)
    adj_full = adj_full + adj_full.T

    return adj_full

def test_encode_decode_adj():
######## code test ###########
    G = nx.ladder_graph(5)
    G = nx.grid_2d_graph(20,20)
    G = nx.ladder_graph(200)
    G = nx.karate_club_graph()
    G = nx.connected_caveman_graph(2,3)
    print(G.number_of_nodes())
    
    adj = np.asarray(nx.to_numpy_matrix(G))
    G = nx.from_numpy_matrix(adj)
    #
    start_idx = np.random.randint(adj.shape[0])
    x_idx = np.array(bfs_seq(G, start_idx))
    adj = adj[np.ix_(x_idx, x_idx)]
    
    print('adj\n',adj)
    adj_output = encode_adj(adj,max_prev_node=5)
    print('adj_output\n',adj_output)
    adj_recover = decode_adj(adj_output,max_prev_node=5)
    print('adj_recover\n',adj_recover)
    print('error\n',np.amin(adj_recover-adj),np.amax(adj_recover-adj))
    
    
    adj_output = encode_adj_flexible(adj)
    for i in range(len(adj_output)):
        print(len(adj_output[i]))
    adj_recover = decode_adj_flexible(adj_output)
    print(adj_recover)
    print(np.amin(adj_recover-adj),np.amax(adj_recover-adj))



def encode_adj_full(adj):
    '''
    return a n-1*n-1*2 tensor, the first dimension is an adj matrix, the second show if each entry is valid
    :param adj: adj matrix
    :return:
    '''
    # pick up lower tri
    adj = np.tril(adj, k=-1)
    n = adj.shape[0]
    adj = adj[1:n, 0:n-1]
    adj_output = np.zeros((adj.shape[0],adj.shape[1],2))
    adj_len = np.zeros(adj.shape[0])

    for i in range(adj.shape[0]):
        non_zero = np.nonzero(adj[i,:])[0]
        input_start = np.amin(non_zero)
        input_end = i + 1
        adj_slice = adj[i, input_start:input_end]
        # write adj
        adj_output[i,0:adj_slice.shape[0],0] = adj_slice[::-1] # put in reverse order
        # write stop token (if token is 0, stop)
        adj_output[i,0:adj_slice.shape[0],1] = 1 # put in reverse order
        # write sequence length
        adj_len[i] = adj_slice.shape[0]

    return adj_output,adj_len

def decode_adj_full(adj_output):
    '''
    return an adj according to adj_output
    :param
    :return:
    '''
    # pick up lower tri
    adj = np.zeros((adj_output.shape[0]+1,adj_output.shape[1]+1))

    for i in range(adj_output.shape[0]):
        non_zero = np.nonzero(adj_output[i,:,1])[0] # get valid sequence
        input_end = np.amax(non_zero)
        adj_slice = adj_output[i, 0:input_end+1, 0] # get adj slice
        # write adj
        output_end = i+1
        output_start = i+1-input_end-1
        adj[i+1,output_start:output_end] = adj_slice[::-1] # put in reverse order
    adj = adj + adj.T
    return adj

def test_encode_decode_adj_full():
########### code test #############
    # G = nx.ladder_graph(10)
    G = nx.karate_club_graph()
    # get bfs adj
    adj = np.asarray(nx.to_numpy_matrix(G))
    G = nx.from_numpy_matrix(adj)
    start_idx = np.random.randint(adj.shape[0])
    x_idx = np.array(bfs_seq(G, start_idx))
    adj = adj[np.ix_(x_idx, x_idx)]
    
    adj_output, adj_len = encode_adj_full(adj)
    print('adj\n',adj)
    print('adj_output[0]\n',adj_output[:,:,0])
    print('adj_output[1]\n',adj_output[:,:,1])
    # print('adj_len\n',adj_len)
    
    adj_recover = decode_adj_full(adj_output)
    print('adj_recover\n', adj_recover)
    print('error\n',adj_recover-adj)
    print('error_sum\n',np.amax(adj_recover-adj), np.amin(adj_recover-adj))






########## use pytorch dataloader
class Graph_sequence_sampler_pytorch(torch.utils.data.Dataset):
    def __init__(self, G_list, max_num_node=None, max_prev_node=None, iteration=20000):
        self.adj_all, self.node_num_all, self.raw_node_f_all, self.child_num, self.len_all = \
            [], [], [], [], []
        self.DFS_first_node = []
        for i, G in enumerate(G_list):
            # add node_type_feature_matrix and edge_type_feature_matrix
            for node in G.nodes():
                if G.nodes[node]['f1'] == 1:
                    first_n = list(G.nodes).index(node)
                    self.DFS_first_node.append(first_n)

            self.adj_all.append(np.array(nx.to_numpy_matrix(G)))

            node_idx_global = np.asarray(list(G.nodes))
            self.node_num_all.append(node_idx_global)
            child_dic = {}
            max_child_node = 0
            for node in G.nodes():
                if G.nodes[node]['f1'] == 1:
                    num_neighbors = len(list(G.neighbors(node)))
                    child_dic[list(G.nodes).index(node)] = num_neighbors
                    max_child_node = max(num_neighbors,max_child_node)
                else:
                    num_neighbors = len(list(G.neighbors(node)))
                    child_dic[list(G.nodes).index(node)] = num_neighbors-1
                    max_child_node = max(num_neighbors-1, max_child_node)
            self.max_child_node = max_child_node
            self.child_num.append(child_dic)

            self.raw_node_f_all.append(dict(G.nodes._nodes))

            self.len_all.append(G.number_of_nodes())

        self.n = max(self.len_all)

        # self.max_prev_node = max_prev_node

        # # sort Graph in descending order
        # len_batch_order = np.argsort(np.array(self.len_all))[::-1]
        # self.len_all = [self.len_all[i] for i in len_batch_order]
        # self.adj_all = [self.adj_all[i] for i in len_batch_order]
    def __len__(self):
        return len(self.adj_all)
    def __getitem__(self, idx):
        adj_copy = self.adj_all[idx].copy()
        node_dict = self.raw_node_f_all[idx].copy()
        child_num = self.child_num[idx].copy()
        node_num_list = self.node_num_all[idx].copy()

        node_feature = self.construct_raw_node_f(node_dict,node_num_list)
        number_of_children = self.construct_node_child(child_num, node_num_list)

        len_batch = adj_copy.shape[0]

        adj_copy_matrix = np.asmatrix(adj_copy)
        G =nx.from_numpy_matrix((adj_copy_matrix))
        start_idx = self.DFS_first_node[idx]
        x_idx = np.array(dfs_seq(G,start_idx))

        node_feature = node_feature[x_idx,:]
        number_of_children = number_of_children[x_idx,:]

        x_batch = np.zeros((self.n,node_feature.shape[1]+self.max_child_node))
        x_batch[0:node_feature.shape[0],:] = np.concatenate((node_feature, number_of_children), axis=1)

        return {'x':x_batch,'len':len_batch}

    def construct_raw_node_f(self, node_dict, node_num_list):
        node_attr_list = list(next(iter(node_dict.values())).keys())
        N, NF = len(node_dict), len(node_attr_list)
        offset = min(node_num_list)
        raw_node_f = np.zeros(shape=(N, NF))  # pad 0 for small graphs
        # idx_list = list(range(N))
        for node, f_dict in node_dict.items():
            if node in node_num_list:
                raw_node_f[node - offset] = np.asarray(list(f_dict.values()))  # 0-indexed

        raw_node_f = raw_node_f[node_num_list - offset, :]
        # raw_node_f[:,-1] = 1
        return raw_node_f
    def construct_node_child(self, node_child, node_num_list):
        node_child_num_list = list(node_child.values())
        N,CN = len(node_child), max(node_child_num_list)
        offset = min(node_num_list)
        number_of_child = np.zeros(shape=(N,CN))
        for node, child in node_child.items():
            if node in node_num_list and child-1 > -1:
                number_of_child[node-offset][child-1] = 1
        number_of_child = number_of_child[node_num_list - offset, :]
        return number_of_child
