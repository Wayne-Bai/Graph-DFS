import networkx as nx
import numpy as np

from utils import *
from data import *
from datat_process import Graph_load_batch as ast_graph_load_batch

def create(args):
### load datasets
    graphs=[]
    # synthetic graphs
    if args.graph_type == 'AST' or args.graph_type == '200Graphs':
        graphs, rule_matrix = ast_graph_load_batch(min_num_nodes=1, name=args.graph_type)
        # update edge_feature_output_dim
        if not args.max_node_feature_num:
            # print(type(graphs[1].nodes._nodes), graphs[1].nodes._nodes.keys())
            args.max_node_feature_num = len(list(graphs[1].nodes._nodes._atlas[1].keys()))  # now equals to 28
        if not args.max_child_node:
            max_child_node = 0
            for i in range(len(graphs)):
                for node in graphs[1].nodes():
                    temp_max_child_node = graphs[1].degree(node)
                    max_child_node = max(temp_max_child_node, max_child_node)
            args.max_child_node = max_child_node

    return graphs


