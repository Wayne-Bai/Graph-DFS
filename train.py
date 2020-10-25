import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from sklearn.decomposition import PCA
import logging
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from time import gmtime, strftime
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from random import shuffle
import pickle
from tensorboard_logger import configure, log_value
import scipy.misc
import time as tm

from utils import *
from model import *
from data import *
from args import Args
import create_graphs

args = Args()

# show all value in the matrix
torch.set_printoptions(profile='full', threshold=np.inf)
np.set_printoptions(threshold=np.inf)


def train_rnn_epoch(epoch, args, rnn, output, data_loader,
                    optimizer_rnn, optimizer_output,
                    scheduler_rnn, scheduler_output,
                    node_f_gen=None, edge_f_gen=None):
    rnn.train()

    loss_sum = 0
    for batch_idx, data in enumerate(data_loader):
        rnn.zero_grad()
        x_unsorted = data['x'].float() # N * (NF + max CN)
        y_len_unsorted = data['len']
        y_len_max = max(y_len_unsorted)
        x_unsorted = x_unsorted[:, 0:y_len_max, :]
        rnn.hidden = rnn.init_hidden(batch_size=x_unsorted.size(0))

        y_len, sort_index = torch.sort(y_len_unsorted, 0, descending=True)
        y_len = y_len.numpy().tolist()

        x = torch.index_select(x_unsorted,0,sort_index)
        output_x = Variable(np.argmax(x,axis=-1)).cuda()
        x = Variable(x).cuda()

        h = rnn(x, pack=True, input_len = y_len)

        x_pred = node_f_gen(h)

        loss = new_cross_entropy(x_pred, output_x, if_CE=True, mask_len=y_len)
        loss.backward()

        optimizer_rnn.step()
        scheduler_rnn.step()

        if epoch % args.epochs_log == 0 and batch_idx == 0:  # only output first batch's statistics
            print('Epoch: {}/{}, train loss: {:.6f}, graph type: {}, num_layer: {}, hidden: {}'.format(
                epoch, args.epochs, loss.data, args.graph_type, args.num_layers, args.hidden_size_rnn))

        # logging
        log_value('loss_' + args.fname, loss.data, epoch * args.batch_ratio + batch_idx)
        feature_dim = x.size(1) * x.size(2)
        loss_sum += loss.data * feature_dim

    return loss_sum / (batch_idx + 1)


def test_rnn_epoch(epoch, args, rnn, output, node_f_gen=None, edge_f_gen=None, test_batch_size=16, test_set=None):
    flag_node_f_gen = False
    if node_f_gen:
        flag_node_f_gen = True
    rnn.hidden = rnn.init_hidden(test_batch_size)
    rnn.eval()
    if flag_node_f_gen:
        node_f_gen.eval()

    max_num_node = int(args.max_num_node)
    x_step = Variable(torch.ones(test_batch_size,1,args.max_node_feature_num+args.max_child_node))
    for i in range(max_num_node):
        h = rnn(x_step)
        x_pred_step = node_f_gen(h)





########### train function for LSTM + VAE
def train(args, dataset_train, rnn, output, node_f_gen=None, edge_f_gen=None, test_set=None):
    # check if load existing model
    if args.load:
        fname = args.model_save_path + args.fname + 'lstm_' + str(args.load_epoch) + '.dat'
        rnn.load_state_dict(torch.load(fname))
        fname = args.model_save_path + args.fname + 'output_' + str(args.load_epoch) + '.dat'
        output.load_state_dict(torch.load(fname))

        args.lr = 0.00001
        epoch = args.load_epoch
        print('model loaded!, lr: {}'.format(args.lr))
    else:
        epoch = 1

    # initialize optimizer
    optimizer_rnn = optim.Adam(list(rnn.parameters()), lr=args.lr)
    optimizer_output = optim.Adam(list(output.parameters()), lr=args.lr)

    scheduler_rnn = MultiStepLR(optimizer_rnn, milestones=args.milestones, gamma=args.lr_rate)
    scheduler_output = MultiStepLR(optimizer_output, milestones=args.milestones, gamma=args.lr_rate)

    # start main loop
    time_all = np.zeros(args.epochs)
    while epoch <= args.epochs:
        time_start = tm.time()
        # train
        if 'GraphRNN_VAE' in args.note:
            train_vae_epoch(epoch, args, rnn, output, dataset_train,
                            optimizer_rnn, optimizer_output,
                            scheduler_rnn, scheduler_output)
        elif 'GraphRNN_MLP' in args.note:
            train_mlp_epoch(epoch, args, rnn, output, dataset_train,
                            optimizer_rnn, optimizer_output,
                            scheduler_rnn, scheduler_output)
        elif 'GraphRNN_RNN' in args.note:
            train_rnn_epoch(epoch, args, rnn, output, dataset_train,
                            optimizer_rnn, optimizer_output,
                            scheduler_rnn, scheduler_output,
                            node_f_gen, edge_f_gen)
        time_end = tm.time()
        time_all[epoch - 1] = time_end - time_start
        # test
        if epoch % args.epochs_test == 0 and epoch >= args.epochs_test_start:
            for sample_time in range(1, 4):
                G_pred = []
                G_test = []
                while len(G_pred) < args.test_total_size:
                    if 'GraphRNN_VAE' in args.note:
                        G_pred_step = test_vae_epoch(epoch, args, rnn, output, test_batch_size=args.test_batch_size,
                                                     sample_time=sample_time)
                    elif 'GraphRNN_MLP' in args.note:
                        G_pred_step = test_mlp_epoch(epoch, args, rnn, output, test_batch_size=args.test_batch_size,
                                                     sample_time=sample_time)
                    elif 'GraphRNN_RNN' in args.note:
                        if args.if_test_use_groundtruth:
                            G_pred_step, G_test_step = test_rnn_epoch(epoch, args, rnn, output, node_f_gen,
                                                                      test_batch_size=args.test_batch_size,
                                                                      test_set=test_set)
                        else:
                            G_pred_step = test_rnn_epoch(epoch, args, rnn, output, node_f_gen,
                                                         test_batch_size=args.test_batch_size, test_set=test_set)
                    if args.if_test_use_groundtruth:
                        G_pred.extend(G_pred_step)
                        G_test.extend(G_test_step)
                    else:
                        G_pred.extend(G_pred_step)
                # save test graphs
                if args.if_test_use_groundtruth:
                    fname = args.graph_save_path + args.fname_pred + str(epoch) + '_' + 'test' + '_' + str(
                        sample_time) + '.dat'
                    save_graph_list(G_test, fname)
                # save graphs
                fname = args.graph_save_path + args.fname_pred + str(epoch) + '_' + str(sample_time) + '.dat'
                save_graph_list(G_pred, fname)
                if 'GraphRNN_RNN' in args.note:
                    break
            print('test done, graphs saved')

        # save model checkpoint
        if args.save:
            if epoch % args.epochs_save == 0:
                fname = args.model_save_path + args.fname + 'lstm_' + str(epoch) + '.dat'
                torch.save(rnn.state_dict(), fname)
                fname = args.model_save_path + args.fname + 'output_' + str(epoch) + '.dat'
                torch.save(output.state_dict(), fname)
        epoch += 1
    np.save(args.timing_save_path + args.fname, time_all)

