import config
import models
import numpy as np
import os
import argparse
import torch

DOCRED = 'docred'

data_set = DOCRED
BATCH_SIZE = 20
HIDDEN_DIM = 120
LR = 1e-3
MAX_EPOCH = 200
SEED = 0 #random.randint(0, 10000)
NAME = 'Struct'
EMB_DIM = 100
DECAY_RATE = 0.95

parser = argparse.ArgumentParser()

# configurations for data
parser.add_argument('--data_path', type=str, default='./prepro_data')
parser.add_argument('--model_name', type = str, default = 'LSR', help = 'name of the model')
parser.add_argument('--train_prefix', type = str, default = 'dev_train')
parser.add_argument('--test_prefix', type = str, default = 'dev_dev')

# configurations for model
parser.add_argument('--finetune_emb', type=int, default=False, help='Fine tune word Embedding.')
parser.add_argument('--rnn_layers', type=int, default=1, help='Number of RNN layers.')
parser.add_argument('--use_struct_att', type=bool, default=False)
parser.add_argument('--use_reasoning_block', type=bool, default=True)
parser.add_argument('--reasoner_layer_first', type=int, default=3, help='Num of the second sublayers in dcgcn block.')
parser.add_argument('--reasoner_layer_second', type=int, default=4, help='Num of the second sublayers in dcgcn block.')


# configurations for dimension
parser.add_argument('--emb_dim', type=int, default=EMB_DIM, help='Word embedding dimension.')
parser.add_argument('--coref_dim', type=int, default=20, help='NER embedding dimension.')
parser.add_argument('--ner_dim', type=int, default=20, help='Coreference embedding dimension.')
parser.add_argument('--pos_dim', type=int, default=20, help='POS embedding dimension.')
parser.add_argument('--hidden_dim', type=int, default=HIDDEN_DIM, help='RNN hidden state size.')

# configurations for dropout
parser.add_argument('--dropout_emb', type=float, default=0.2, help='embedding dropout rate.')
parser.add_argument('--dropout_rnn', type=float, default=0.2, help='embedding dropout rnn.')
parser.add_argument('--dropout_rate', type=float, default=0.2, help='dropout rate for FF network.')
parser.add_argument('--dropout_gcn', type=float, default=0.3, help='GCN  dropout rate.')

# configurations for training
parser.add_argument('--lr', type=float, default=LR, help='Applies to sgd and adagrad.')
parser.add_argument('--lr_decay', type=float, default=DECAY_RATE, help='Learning rate decay rate.')
parser.add_argument('--decay_epoch', type=int, default=7, help='Decay learning rate after this epoch.')
parser.add_argument('--evaluate_epoch', type=int, default=30, help='Evaluate after this epoch.')
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

# others
parser.add_argument('--seed', type=int, default=SEED)
parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available())
parser.add_argument('--cpu', action='store_true', help='Ignore CUDA.')
parser.add_argument('--input_theta', type = float, default = -1)
parser.add_argument('--save_name', type=str,
					default= NAME + "_DIM_"+ str(EMB_DIM) +"_HIDDEN_" + str(HIDDEN_DIM) + "_" + data_set + "_LR_" + str(
						LR) + "_DECAY_" + str(DECAY_RATE)  + "_BATCHSIZE_" + str(BATCH_SIZE) +"_SEED_"+str(SEED))

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
