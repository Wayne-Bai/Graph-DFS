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
        if args.dataset_type == "2" or args.dataset_type == '500-10':
            args.max_prev_node = 6
        elif args.dataset_type == "54" or args.dataset_type == '2-30':
            args.max_prev_node = 20
        elif args.dataset_type == "9":
            args.max_prev_node = 150
        elif args.dataset_type == "500":
            args.max_prev_node = 40
        elif args.dataset_type == "50-200":
            args.max_prev_node = 150
        elif args.dataset_type == "POC":
            args.max_prev_node = 90
        elif args.dataset_type == 'UAF':
            args.max_prev_node = 50
        elif args.dataset_type == 'TC':
            args.max_prev_node = 80

    return graphs


