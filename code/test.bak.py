import config
import models
import numpy as np
import os
import argparse
import torch

parser = argparse.ArgumentParser()

DOCRED = 'docred'


DEBUG_MODE = "DEBUG"
RUNNING_MODE = 'RUN'
CMU = 'cmu'

data_set = DOCRED
BATCH_SIZE = 20
WORKING_MODE = RUNNING_MODE
HIDDEN_DIM = 120
LR = 1e-3
STRUCT_MODEL = CMU
MAX_EPOCH = 200
SEED = 0 #random.randint(0, 10000)
SPLIT = True
NAME = 'Struct'
STRUCT_LAYER = 2
BLOCK = 2
REFINE = True
VANI = False
GCGNN = False

parser.add_argument('--data_path', type=str, default='./prepro_data')
# dataset related
parser.add_argument('--use_docred', type=bool, default=True)
EMB_DIM = 100
DECAY_RATE = 0.95
# configuration for data
parser.add_argument('--model_name', type = str, default = 'LSR', help = 'name of the model')
parser.add_argument('--train_prefix', type = str, default = 'dev_train')
parser.add_argument('--test_prefix', type = str, default = 'dev_dev')
parser.add_argument('--working_mode', type=str, default=WORKING_MODE)

# configuration for working mode
parser.add_argument('--multiPhase', type=bool, default=True)

parser.add_argument('--use_naacl_model', type=bool, default=False)
parser.add_argument('--use_sent_dep_tree', type=bool, default=False)
parser.add_argument('--use_sdp_node', type=bool, default=True)
parser.add_argument('--use_sent_node', type=bool, default=False)
parser.add_argument('--use_sdp_sent_node', type=bool, default=False)
parser.add_argument('--use_gcn_variant', type=bool, default=True)
parser.add_argument('--use_dcgcn_struct_node', type=bool, default=False)
parser.add_argument('--use_e_m_sdp_adj', type=bool, default=False)

parser.add_argument('--use_struct_att_block_e', type=bool, default=False)
parser.add_argument('--use_struct_att_block_r', type=bool, default=True)
parser.add_argument('--use_node_emb', type=bool, default=False)

parser.add_argument('--vanlila_gcn', type=bool, default= VANI)

parser.add_argument('--refinement', type=bool, default= REFINE)
parser.add_argument('--split_vector', type=bool, default= SPLIT, help='whether split the input or not in struct attention module.')

#general hyper-parameters
parser.add_argument('--emb_dim', type=int, default=EMB_DIM, help='Word embedding dimension.')
parser.add_argument('--finetune_emb', type=int, default=False, help='Fine tune word Embedding.')

parser.add_argument('--dropout_emb', type=float, default=0.2, help='embedding dropout rate.')
parser.add_argument('--dropout_rnn', type=float, default=0.2, help='embedding dropout rate.')
parser.add_argument('--dropout_rate', type=float, default=0.2, help='dropout rate for FF network.')
parser.add_argument('--dropout_in', type=float, default=0.3, help='DCGCN Input dropout rate.')

parser.add_argument('--coref_dim', type=int, default=20, help='NER embedding dimension.')
parser.add_argument('--ner_dim', type=int, default=20, help='NER embedding dimension.')
parser.add_argument('--pos_dim', type=int, default=20, help='POS embedding dimension.')

parser.add_argument('--hidden_dim', type=int, default=HIDDEN_DIM, help='RNN hidden state size.')
parser.add_argument('--topn', type=int, default=1e10, help='Only finetune top N word embeddings.')
parser.add_argument('--lower', dest='lower', action='store_true', help='Lowercase all words.')
parser.add_argument('--no-lower', dest='lower', action='store_false')
parser.set_defaults(lower=False)

parser.add_argument('--rnn_layers', type=int, default=1, help='Number of RNN layers.')
parser.add_argument('--lr', type=float, default=LR, help='Applies to sgd and adagrad.')
parser.add_argument('--lr_decay', type=float, default=DECAY_RATE, help='Learning rate decay rate.')
parser.add_argument('--decay_epoch', type=int, default=7, help='Decay learning rate after this epoch.')
parser.add_argument('--evaluate_epoch', type=int, default=11, help='Decay learning rate after this epoch.')
parser.add_argument('--optim', choices=['sgd', 'adagrad', 'adam', 'adamax'], default='adam', help='Optimizer: sgd, adagrad, adam or adamax.')
parser.add_argument('--num_epoch', type=int, default=MAX_EPOCH, help='Number of total training epochs.')
parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='Training batch size.')
parser.add_argument('--max_grad_norm', type=float, default=5.0, help='Gradient clipping.')
parser.add_argument('--log_step', type=int, default=30, help='Print log every k steps.')
parser.add_argument('--log', type=str, default='logs.txt', help='Write training log to file.')
parser.add_argument('--save_epoch', type=int, default=100, help='Save model checkpoints every k epochs.')
parser.add_argument('--save_dir', type=str, default='./saved_models', help='Root dir for saving models.')
parser.add_argument('--id', type=str, default='00', help='Model ID under which to save models.')
parser.add_argument('--info', type=str, default='', help='Optional info for the experiment.')

#GCN/DCGCN DropOut
parser.add_argument('--dropout_out', type=float, default=0.3, help='GCN layer dropout rate.')

# hyper-parameters for Dependency based GCN
parser.add_argument('--dep_tree_layers', type=int, default=2, help="dep tree gcn layers number")
parser.add_argument('--block_num', type=int, default=BLOCK)
parser.add_argument('--sublayer_first', type=int, default=4, help='Num of the first sublayers in dcgcn block.')
parser.add_argument('--sublayer_second', type=int, default=4, help='Num of the second sublayers in dcgcn block.')

# struct attention
parser.add_argument('--struct_layers_e', type=int, default=1, help='Num of the struct attention.')
parser.add_argument('--struct_layers_r', type=int, default=STRUCT_LAYER, help='Num of the struct attention.') # ivan
parser.add_argument('--struct_sublayer_e_first', type=int, default=1, help='Num of the first sublayers in dcgcn block.')
parser.add_argument('--struct_sublayer_r_first', type=int, default=3, help='Num of the second sublayers in dcgcn block.')
parser.add_argument('--struct_sublayer_r_second', type=int, default=4, help='Num of the second sublayers in dcgcn block.')
parser.add_argument('--struct_sublayer_r_rule', type=int, default=3, help='Num of the layers for the rule-based DCGCN.')
parser.add_argument('--struct_no_split', type=bool, default=False, help="split for struct attention")

# mutihead
parser.add_argument('--multihead_block_gcn_layers', type=int, default=3, help="the number of layers for multihead block in doc-level reasoning")
parser.add_argument('--head', type=int, default=3, help="multihead number that are used for sentence-level adj")

# others
parser.add_argument('--seed', type=int, default=SEED)
parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available())
parser.add_argument('--cpu', action='store_true', help='Ignore CUDA.')

# Depreciated! this parameters are discarded, please ingore them
parser.add_argument('--binary', type=bool, default=False)
parser.add_argument('--use_warmup', type = bool, default = False)
parser.add_argument('--use_half', type = bool, default = False)
parser.add_argument('--use_gcn', type=bool, default=False)
parser.add_argument('--use_atten_structure_encoding', type=bool, default= False)
parser.add_argument('--use_atten_structure_reasoning', type=bool, default=False)

# Depreciated! hyper-parameters for pair-wised ranking loss # only for binary mode
parser.add_argument('--mPos', type=float, default=2.5, help="margin for positive instances")
parser.add_argument('--mNeg', type=float, default=0.5, help="margin for negative instances")
parser.add_argument('--gamma', type=float, default=2, help="scaling factor")

# Depreciated! hyper-parameters for GCN
parser.add_argument('--vanila_layers', type=int, default=4, help="gcn layers number")
parser.add_argument('--atten_layers', type=int, default=2, help="gcn layers number")

parser.add_argument('--load', dest='load', action='store_true', help='Load pretrained model.')
parser.add_argument('--model_file', type=str, help='Filename of the pretrained model.')
parser.add_argument('--input_theta', type = float, default = -1)

parser.add_argument('--save_name', type=str,
					default= NAME + "_DIM_"+ str(EMB_DIM) +"_HIDDEN_" + str(HIDDEN_DIM) + "_" + data_set + "_LR_" + str(
						LR) + "_DECAY_" + str(DECAY_RATE) + "_" + str(STRUCT_MODEL) + "_BATCHSIZE_" + str(BATCH_SIZE) + "_LAYERS_"+str(STRUCT_LAYER)+ "_BLOCK_"+str(BLOCK)+"_SEED_"+str(SEED) + "_SPLIT_"+str(SPLIT)+"_REFINE_"+str(REFINE)+"_VAN_"+str(VANI))

args = parser.parse_args()

def logging(s, print_=True, log_=True):
	if print_:
		print(s)
	if log_:
		with open(os.path.join(os.path.join("log", args.save_name)), 'a+') as f_log:
			f_log.write(s + '\n')


import random

torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

if args.cpu:
    args.cuda = False
elif args.cuda:
    torch.cuda.manual_seed(args.seed)

# init_time = time.time()

def print_config(config):
    info = "Running with the following configs:\n"
    for k,v in vars(config).items():
        info += "\t{} : {}\n".format(k, str(v))
    print("\n" + info + "\n")
    return

print_config(args)

model = {
	'CNN3': models.CNN3,
	'LSTM': models.LSTM,
	'BiLSTM': models.BiLSTM,
	'ContextAware': models.ContextAware,
	'LSR': models.LSR,
}

con = config.Config(args)
con.load_test_data()

import datetime
print(datetime.datetime.now())

con.testall(model[args.model_name], args.save_name, args.input_theta)
