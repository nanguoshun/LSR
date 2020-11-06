# coding: utf-8
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import time
import json
import sklearn.metrics
# import matplotlib
# matplotlib.use('Agg')
import random
import gc
from collections import defaultdict

from operator import add

# BERT_ENCODER = 'bert'
# CHEMICAL_TYPE = 'Chemical'
# GENE_TYPE = 'Gene'
# DISEASE_TYPE = 'Disease'
# DEBUG_MODE = "DEBUG"
# RUNNING_MODE = 'RUN'

MAX_NODE_NUM = 512

IGNORE_INDEX = -100
is_transformer = False

DEBUG_DOC_NO = 60

from utils import torch_utils

def isNaN(num):
    return num != num

class Node:
    def __init__(self, id, v_id, v_no, sent_id, pos_start, pos_end):
        self.id = id
        self.v_id = v_id
        self.v_no = v_no
        self.sent_id = sent_id
        self.pos_start = pos_start
        self.pos_end = pos_end

class Accuracy(object):
    def __init__(self):
        self.correct = 0
        self.total = 0
    def add(self, is_correct):
        self.total += 1
        if is_correct:
            self.correct += 1
    def get(self):
        if self.total == 0:
            return 0.0
        else:
            return float(self.correct) / self.total
    def clear(self):
        self.correct = 0
        self.total = 0

class ConfigBert(object):
    def __init__(self, args):
        self.opt = args
        self.acc_NA = Accuracy()
        self.acc_not_NA = Accuracy()
        self.acc_total = Accuracy()
        self.data_path = args.data_path
        self.use_bag = False
        self.use_gpu = True
        self.is_training = True
        self.max_length = 512
        self.pos_num = 2 * self.max_length
        self.entity_num = self.max_length

        self.relation_num = 97
        self.ner_vocab_len = 13

        self.max_sent_len = 200
        self.max_entity_num = 100
        self.max_sent_num = 30
        self.max_node_num = 200
        self.max_node_per_sent = 40

        self.rnn_hidden = args.hidden_dim # hidden emb dim
        self.coref_size = args.coref_dim # coref emb dim
        self.entity_type_size = args.pos_dim # entity emb dim
        self.max_epoch = args.num_epoch
        self.opt_method = 'Adam'
        self.optimizer = None

        self.checkpoint_dir = './checkpoint'
        self.fig_result_dir = './fig_result'
        self.test_epoch = 1
        self.pretrain_model = None

        self.word_size = 100
        self.epoch_range = None
        self.dropout_rate = args.dropout_rate  # for sequence
        self.keep_prob = 0.8  # for lstm

        self.period = 50
        self.batch_size = args.batch_size
        self.h_t_limit = 1800

        self.test_batch_size = self.batch_size
        self.test_relation_limit = 1800
        self.char_limit = 16
        self.sent_limit = 25
        self.dis2idx = np.zeros((512), dtype='int64')
        self.dis2idx[1] = 1
        self.dis2idx[2:] = 2
        self.dis2idx[4:] = 3
        self.dis2idx[8:] = 4
        self.dis2idx[16:] = 5
        self.dis2idx[32:] = 6
        self.dis2idx[64:] = 7
        self.dis2idx[128:] = 8
        self.dis2idx[256:] = 9
        self.dis_size = 20

        self.train_prefix = args.train_prefix
        self.test_prefix = args.test_prefix

        self.lr = args.lr
        self.decay_epoch = args.decay_epoch

        self.lr_decay = args.lr_decay
        if not os.path.exists("log"):
            os.mkdir("log")

        self.softmax = nn.Softmax(dim=-1)

        self.dropout_emb = args.dropout_emb
        self.dropout_rnn = args.dropout_rnn
        self.dropout_gcn = args.dropout_gcn

        self.max_grad_norm = args.max_grad_norm # gradient clipping
        self.optim = args.optim

        self.use_struct_att = args.use_struct_att

        self.use_reasoning_block = args.use_reasoning_block
        self.reasoner_layer_first = args.reasoner_layer_first
        self.reasoner_layer_second = args.reasoner_layer_second

        self.evaluate_epoch = args.evaluate_epoch
        self.finetune_emb = args.finetune_emb

    def set_data_path(self, data_path):
        self.data_path = data_path
    def set_max_length(self, max_length):
        self.max_length = max_length
        self.pos_num = 2 * self.max_length
    def set_num_classes(self, num_classes):
        self.num_classes = num_classes
    def set_window_size(self, window_size):
        self.window_size = window_size
    def set_word_size(self, word_size):
        self.word_size = word_size
    def set_max_epoch(self, max_epoch):
        self.max_epoch = max_epoch
    def set_batch_size(self, batch_size):
        self.batch_size = batch_size
    def set_opt_method(self, opt_method):
        self.opt_method = opt_method
    def set_drop_prob(self, drop_prob):
        self.drop_prob = drop_prob
    def set_checkpoint_dir(self, checkpoint_dir):
        self.checkpoint_dir = checkpoint_dir
    def set_test_epoch(self, test_epoch):
        self.test_epoch = test_epoch
    def set_pretrain_model(self, pretrain_model):
        self.pretrain_model = pretrain_model
    def set_is_training(self, is_training):
        self.is_training = is_training
    def set_use_bag(self, use_bag):
        self.use_bag = use_bag
    def set_use_gpu(self, use_gpu):
        self.use_gpu = use_gpu
    def set_epoch_range(self, epoch_range):
        self.epoch_range = epoch_range

    def load_train_data(self):
        print("Reading training data...")
        prefix = self.train_prefix

        print ('train', prefix)

        self.data_train_word = np.load(os.path.join(self.data_path, prefix + '_word.npy'))

        # elmo_ids = batch_to_ids(batch_words).cuda()
        self.data_train_pos = np.load(os.path.join(self.data_path, prefix+'_pos.npy'))
        self.data_train_ner = np.load(os.path.join(self.data_path, prefix+'_ner.npy')) # word_embedding
        self.data_train_char = np.load(os.path.join(self.data_path, prefix+'_char.npy'))
        self.data_train_seg = np.load(os.path.join(self.data_path, prefix+'_seg.npy'))
        self.data_train_node_position = np.load(os.path.join(self.data_path, prefix+'_node_position.npy'))

        self.data_train_node_position_sent = np.load(os.path.join(self.data_path, prefix+'_node_position_sent.npy'))

        self.data_train_node_sent_num = np.load(os.path.join(self.data_path, prefix+'_node_sent_num.npy'))

        self.data_train_node_num = np.load(os.path.join(self.data_path, prefix+'_node_num.npy'))
        self.data_train_entity_position = np.load(os.path.join(self.data_path, prefix+'_entity_position.npy'))
        self.train_file = json.load(open(os.path.join(self.data_path, prefix+'.json')))

        self.data_train_sdp_position = np.load(os.path.join(self.data_path, prefix + '_sdp_position.npy'))
        self.data_train_sdp_num = np.load(os.path.join(self.data_path, prefix+'_sdp_num.npy'))
        self.data_train_bert_word = np.load(os.path.join(self.data_path, prefix+'_bert_word.npy'))
        self.data_train_bert_mask = np.load(os.path.join(self.data_path, prefix+'_bert_mask.npy'))
        self.data_train_bert_starts = np.load(os.path.join(self.data_path, prefix+'_bert_starts.npy'))

        self.train_len = ins_num = self.data_train_word.shape[0]

        assert(self.train_len==len(self.train_file))
        print("Finish reading, total reading {} train documetns".format(self.train_len))

        self.train_order = list(range(ins_num))
        self.train_batches = ins_num // self.batch_size
        if ins_num % self.batch_size != 0:
            self.train_batches += 1

    def load_test_data(self):
        print("Reading testing data...")

        self.data_word_vec = np.load(os.path.join(self.data_path, 'vec.npy'))

        print("vocab size is: {}".format(len(self.data_word_vec)))

        self.rel2id = json.load(open(os.path.join(self.data_path, 'rel2id.json')))
        self.id2rel = {v: k for k,v in self.rel2id.items()}

        prefix = self.test_prefix

        print (prefix)
        self.is_test = ('dev_test' == prefix)

        self.data_test_word = np.load(os.path.join(self.data_path, prefix + '_word.npy'))
        self.data_test_pos = np.load(os.path.join(self.data_path, prefix+'_pos.npy'))
        self.data_test_ner = np.load(os.path.join(self.data_path, prefix+'_ner.npy'))
        self.data_test_char = np.load(os.path.join(self.data_path, prefix+'_char.npy'))

        self.data_test_node_position = np.load(os.path.join(self.data_path, prefix+'_node_position.npy'))

        self.data_test_node_position_sent = np.load(os.path.join(self.data_path, prefix+'_node_position_sent.npy'))
        #self.data_test_adj = np.load(os.path.join(self.data_path, prefix+'_adj.npy'))

        self.data_test_node_sent_num = np.load(os.path.join(self.data_path, prefix+'_node_sent_num.npy'))

        self.data_test_node_num = np.load(os.path.join(self.data_path, prefix+'_node_num.npy'))
        self.data_test_entity_position = np.load(os.path.join(self.data_path, prefix+'_entity_position.npy'))
        self.test_file = json.load(open(os.path.join(self.data_path, prefix+'.json')))
        self.data_test_seg = np.load(os.path.join(self.data_path, prefix+'_seg.npy'))
        self.test_len = self.data_test_word.shape[0]

        self.data_test_sdp_position = np.load(os.path.join(self.data_path, prefix + '_sdp_position.npy'))
        self.data_test_sdp_num = np.load(os.path.join(self.data_path, prefix+'_sdp_num.npy'))

        self.data_test_bert_word = np.load(os.path.join(self.data_path, prefix+'_bert_word.npy'))
        self.data_test_bert_mask = np.load(os.path.join(self.data_path, prefix+'_bert_mask.npy'))
        self.data_test_bert_starts = np.load(os.path.join(self.data_path, prefix+'_bert_starts.npy'))

        assert(self.test_len==len(self.test_file))

        print("Finish reading, total reading {} test documetns".format(self.test_len))

        self.test_batches = self.data_test_word.shape[0] // self.test_batch_size
        if self.data_test_word.shape[0] % self.test_batch_size != 0:
            self.test_batches += 1

        self.test_order = list(range(self.test_len))
        self.test_order.sort(key=lambda x: np.sum(self.data_test_word[x] > 0), reverse=True)

    def get_train_batch(self):

        random.shuffle(self.train_order)
        context_idxs = torch.LongTensor(self.batch_size, self.max_length).cuda()
        context_pos = torch.LongTensor(self.batch_size, self.max_length).cuda()
        h_mapping = torch.Tensor(self.batch_size, self.h_t_limit, self.max_length).cuda()
        t_mapping = torch.Tensor(self.batch_size, self.h_t_limit, self.max_length).cuda()
        relation_multi_label = torch.Tensor(self.batch_size, self.h_t_limit, self.relation_num).cuda()
        relation_mask = torch.Tensor(self.batch_size, self.h_t_limit).cuda()
        context_masks = torch.LongTensor(self.test_batch_size, self.max_length).cuda()
        context_starts = torch.LongTensor(self.test_batch_size, self.max_length).cuda()

        pos_idx = torch.LongTensor(self.batch_size, self.max_length).cuda()

        context_ner = torch.LongTensor(self.batch_size, self.max_length).cuda()
        context_char_idxs = torch.LongTensor(self.batch_size, self.max_length, self.char_limit).cuda()

        relation_label = torch.LongTensor(self.batch_size, self.h_t_limit).cuda()

        ht_pair_pos = torch.LongTensor(self.batch_size, self.h_t_limit).cuda()

        context_seg = torch.LongTensor(self.batch_size, self.max_length).cuda()

        node_position_sent = torch.zeros(self.batch_size, self.max_sent_num, self.max_node_per_sent, self.max_sent_len).float()

        #cgnn_adj = torch.zeros(self.batch_size, 5, self.max_length,self.max_length).float() # 5 indicate the rational type in GCGNN
        node_position =  torch.zeros(self.batch_size, self.max_node_num, self.max_length).float().cuda()

        sdp_position = torch.zeros(self.batch_size, self.max_entity_num, self.max_length).float().cuda()
        sdp_num = torch.zeros(self.batch_size, 1).long().cuda()

        node_sent_num =  torch.zeros(self.batch_size, self.max_sent_num).float().cuda()

        entity_position =  torch.zeros(self.batch_size, self.max_entity_num, self.max_length).float().cuda()
        node_num = torch.zeros(self.batch_size, 1).long().cuda()

        for b in range(self.train_batches):

            entity_num = []
            sentence_num = []
            sentence_len = []
            node_num_per_sent = []

            start_id = b * self.batch_size
            cur_bsz = min(self.batch_size, self.train_len - start_id)
            cur_batch = list(self.train_order[start_id: start_id + cur_bsz])
            cur_batch.sort(key=lambda x: np.sum(self.data_train_word[x]>0) , reverse = True)

            for mapping in [h_mapping, t_mapping]:
                mapping.zero_()

            for mapping in [relation_multi_label, relation_mask, pos_idx]:
                mapping.zero_()

            ht_pair_pos.zero_()

            relation_label.fill_(IGNORE_INDEX)

            max_h_t_cnt = 1

            sdp_nums = []

            for i, index in enumerate(cur_batch):
                context_idxs[i].copy_(torch.from_numpy(self.data_train_bert_word[index, :]))
                # context_idxs[i].copy_(torch.from_numpy(self.data_train_word[index, :]))
                context_pos[i].copy_(torch.from_numpy(self.data_train_pos[index, :])) #???
                context_char_idxs[i].copy_(torch.from_numpy(self.data_train_char[index, :]))
                context_ner[i].copy_(torch.from_numpy(self.data_train_ner[index, :]))
                context_seg[i].copy_(torch.from_numpy(self.data_train_seg[index, :]))
                context_masks[i].copy_(torch.from_numpy(self.data_train_bert_mask[index, :]))
                context_starts[i].copy_(torch.from_numpy(self.data_train_bert_starts[index, :]))

                ins = self.train_file[index]
                labels = ins['labels']
                idx2label = defaultdict(list)

                for label in labels:
                    idx2label[(label['h'], label['t'])].append(int(label['r']))

                node_position[i].copy_(torch.from_numpy(self.data_train_node_position[index]))

                node_position_sent[i].copy_(torch.from_numpy(self.data_train_node_position_sent[index]))

                node_sent_num[i].copy_(torch.from_numpy(self.data_train_node_sent_num[index]))

                node_num[i].copy_(torch.from_numpy(self.data_train_node_num[index]))
                entity_position[i].copy_(torch.from_numpy(self.data_train_entity_position[index]))

                entity_num.append(len(ins['vertexSet']))
                sentence_num.append(len(ins['sents']))
                sentence_len.append(max([len(sent) for sent in ins['sents']])) # max sent len of a document
                node_num_per_sent.append(max(node_sent_num[i].tolist()))

                sdp_position[i].copy_(torch.from_numpy(self.data_train_sdp_position[index]))
                sdp_num[i].copy_(torch.from_numpy(self.data_train_sdp_num[index]))

                sdp_no_trucation = sdp_num[i].item()
                if sdp_no_trucation > self.max_entity_num:
                    sdp_no_trucation = self.max_entity_num
                sdp_nums.append(sdp_no_trucation)

                for j in range(self.max_length):
                    if self.data_train_word[index, j]==0:
                        break
                    pos_idx[i, j] = j+1

                train_tripe = list(idx2label.keys())
                for j, (h_idx, t_idx) in enumerate(train_tripe):
                    hlist = ins['vertexSet'][h_idx]
                    tlist = ins['vertexSet'][t_idx]

                    for h in hlist:
                        h_mapping[i, j, h['pos'][0]:h['pos'][1]] = 1.0 / len(hlist) / (h['pos'][1] - h['pos'][0])

                    for t in tlist:
                        t_mapping[i, j, t['pos'][0]:t['pos'][1]] = 1.0 / len(tlist) / (t['pos'][1] - t['pos'][0])

                    label = idx2label[(h_idx, t_idx)]

                    delta_dis = hlist[0]['pos'][0] - tlist[0]['pos'][0]

                    if abs(delta_dis) >= self.max_length: # for gda
                       continue

                    if delta_dis < 0:
                        ht_pair_pos[i, j] = -int(self.dis2idx[-delta_dis])
                    else:
                        ht_pair_pos[i, j] = int(self.dis2idx[delta_dis])

                    for r in label:
                        relation_multi_label[i, j, r] = 1

                    relation_mask[i, j] = 1
                    rt = np.random.randint(len(label))
                    relation_label[i, j] = label[rt]

                lower_bound = len(ins['na_triple'])

                for j, (h_idx, t_idx) in enumerate(ins['na_triple'][:lower_bound], len(train_tripe)):
                    hlist = ins['vertexSet'][h_idx]
                    tlist = ins['vertexSet'][t_idx]


                    for h in hlist:
                        h_mapping[i, j, h['pos'][0]:h['pos'][1]] = 1.0 / len(hlist) / (h['pos'][1] - h['pos'][0])

                    for t in tlist:
                        t_mapping[i, j, t['pos'][0]:t['pos'][1]] = 1.0 / len(tlist) / (t['pos'][1] - t['pos'][0])

                    relation_multi_label[i, j, 0] = 1
                    relation_label[i, j] = 0
                    relation_mask[i, j] = 1
                    delta_dis = hlist[0]['pos'][0] - tlist[0]['pos'][0]

                    if abs(delta_dis) >= self.max_length:#for gda
                       continue

                    if delta_dis < 0:
                        ht_pair_pos[i, j] = -int(self.dis2idx[-delta_dis])
                    else:
                        ht_pair_pos[i, j] = int(self.dis2idx[delta_dis])

                max_h_t_cnt = max(max_h_t_cnt, len(train_tripe) + lower_bound)

            input_lengths = (context_idxs[:cur_bsz] > 0).long().sum(dim=1)
            max_c_len = int(input_lengths.max()) # max length of a document

            entity_mention_num = list(map(add, entity_num, node_num[:cur_bsz].squeeze(1).tolist()))
            max_sdp_num = max(sdp_nums)
            all_node_num = list(map(add, sdp_nums , entity_mention_num))

            max_entity_num = max(entity_num)
            max_sentence_num = max(sentence_num)
            b_max_mention_num = int(node_num[:cur_bsz].max()) #- max_entity_num - max_sentence_num
            all_node_num = torch.LongTensor(all_node_num)

            yield {'context_idxs': context_idxs[:cur_bsz, :max_c_len].contiguous(),
                   'context_masks': context_masks[:cur_bsz, :max_c_len].contiguous(), #Ivan
                   'context_starts': context_starts[:cur_bsz, :max_c_len].contiguous(),
                   'context_pos': context_pos[:cur_bsz, :max_c_len].contiguous(),
                   'h_mapping': h_mapping[:cur_bsz, :max_h_t_cnt, :max_c_len],
                   't_mapping': t_mapping[:cur_bsz, :max_h_t_cnt, :max_c_len],
                   'relation_label': relation_label[:cur_bsz, :max_h_t_cnt].contiguous(),
                   'input_lengths' : input_lengths,
                   'pos_idx': pos_idx[:cur_bsz, :max_c_len].contiguous(),
                   'relation_multi_label': relation_multi_label[:cur_bsz, :max_h_t_cnt],
                   'relation_mask': relation_mask[:cur_bsz, :max_h_t_cnt],
                   'context_ner': context_ner[:cur_bsz, :max_c_len].contiguous(),
                   'context_char_idxs': context_char_idxs[:cur_bsz, :max_c_len].contiguous(),
                   'ht_pair_pos': ht_pair_pos[:cur_bsz, :max_h_t_cnt],
                   'context_seg': context_seg[:cur_bsz, :max_c_len].contiguous(),
                   'node_position': node_position[:cur_bsz, :b_max_mention_num, :max_c_len].contiguous(),
                   'node_sent_num': node_sent_num[:cur_bsz, :max_sentence_num].contiguous(),
                   'entity_position': entity_position[:cur_bsz, :max_entity_num, :max_c_len].contiguous(),
                   'all_node_num':all_node_num,
                   'entity_num': entity_num,
                   'sent_num': sentence_num,
                   'sdp_position': sdp_position[:cur_bsz, :max_sdp_num, :max_c_len].contiguous(),
                   'sdp_num': sdp_nums,
                   }

    def get_test_batch(self):
        context_idxs = torch.LongTensor(self.test_batch_size, self.max_length).cuda()
        context_pos = torch.LongTensor(self.test_batch_size, self.max_length).cuda()
        h_mapping = torch.Tensor(self.test_batch_size, self.test_relation_limit, self.max_length).cuda()
        t_mapping = torch.Tensor(self.test_batch_size, self.test_relation_limit, self.max_length).cuda()
        context_ner = torch.LongTensor(self.test_batch_size, self.max_length).cuda()
        context_char_idxs = torch.LongTensor(self.test_batch_size, self.max_length, self.char_limit).cuda()
        relation_mask = torch.Tensor(self.test_batch_size, self.h_t_limit).cuda()
        ht_pair_pos = torch.LongTensor(self.test_batch_size, self.h_t_limit).cuda()
        context_seg = torch.LongTensor(self.batch_size, self.max_length).cuda()
        context_masks = torch.LongTensor(self.batch_size, self.max_length).cuda()
        context_starts = torch.LongTensor(self.batch_size, self.max_length).cuda()

        node_position_sent =  torch.zeros(self.batch_size, self.max_sent_num, self.max_node_per_sent, self.max_sent_len).float()

        node_position =  torch.zeros(self.batch_size, self.max_node_num, self.max_length).float().cuda()
        entity_position =  torch.zeros(self.batch_size, self.max_entity_num, self.max_length).float().cuda()
        node_num = torch.zeros(self.batch_size, 1).long().cuda()

        node_sent_num =  torch.zeros(self.batch_size, self.max_sent_num).float().cuda()

        sdp_position = torch.zeros(self.batch_size, self.max_entity_num, self.max_length).float().cuda()
        sdp_num = torch.zeros(self.batch_size, 1).long().cuda()

        for b in range(self.test_batches):

            entity_num = []
            sentence_num = []
            sentence_len = []
            node_num_per_sent = []

            start_id = b * self.test_batch_size
            cur_bsz = min(self.test_batch_size, self.test_len - start_id)
            cur_batch = list(self.test_order[start_id : start_id + cur_bsz])

            for mapping in [h_mapping, t_mapping, relation_mask]:
                mapping.zero_()

            ht_pair_pos.zero_()

            max_h_t_cnt = 1

            cur_batch.sort(key=lambda x: np.sum(self.data_test_word[x]>0) , reverse = True)

            labels = []

            L_vertex = []
            titles = []
            indexes = []
            sdp_nums = []

            vertexSets = []

            for i, index in enumerate(cur_batch):
                context_idxs[i].copy_(torch.from_numpy(self.data_test_bert_word[index, :]))
                # context_idxs[i].copy_(torch.from_numpy(self.data_test_word[index, :]))
                context_pos[i].copy_(torch.from_numpy(self.data_test_pos[index, :]))
                context_char_idxs[i].copy_(torch.from_numpy(self.data_test_char[index, :]))
                context_ner[i].copy_(torch.from_numpy(self.data_test_ner[index, :]))
                context_seg[i].copy_(torch.from_numpy(self.data_test_seg[index, :]))
                context_masks[i].copy_(torch.from_numpy(self.data_test_bert_mask[index, :]))
                context_starts[i].copy_(torch.from_numpy(self.data_test_bert_starts[index, :]))

                idx2label = defaultdict(list)
                ins = self.test_file[index]

                for label in ins['labels']:
                    idx2label[(label['h'], label['t'])].append(label['r'])

                node_position[i].copy_(torch.from_numpy(self.data_test_node_position[index]))
                node_position_sent[i].copy_(torch.from_numpy(self.data_test_node_position_sent[index]))

                node_sent_num[i].copy_(torch.from_numpy(self.data_test_node_sent_num[index]))

                node_num[i].copy_(torch.from_numpy(self.data_test_node_num[index]))
                entity_position[i].copy_(torch.from_numpy(self.data_test_entity_position[index]))
                entity_num.append(len(ins['vertexSet']))
                sentence_num.append(len(ins['sents']))
                sentence_len.append(max([len(sent) for sent in ins['sents']])) # max sent len of a document
                node_num_per_sent.append(max(node_sent_num[i].tolist()))

                sdp_position[i].copy_(torch.from_numpy(self.data_test_sdp_position[index]))
                sdp_num[i].copy_(torch.from_numpy(self.data_test_sdp_num[index]))

                sdp_no_trucation = sdp_num[i].item()
                if sdp_no_trucation > self.max_entity_num:
                    sdp_no_trucation = self.max_entity_num
                sdp_nums.append(sdp_no_trucation)

                L = len(ins['vertexSet'])
                titles.append(ins['title'])

                vertexSets.append(ins['vertexSet'])

                j = 0
                for h_idx in range(L):
                    for t_idx in range(L):
                        if h_idx != t_idx:
                            hlist = ins['vertexSet'][h_idx]
                            tlist = ins['vertexSet'][t_idx]

                            for h in hlist:
                                h_mapping[i, j, h['pos'][0]:h['pos'][1]] = 1.0 / len(hlist) / (h['pos'][1] - h['pos'][0])
                            for t in tlist:
                                t_mapping[i, j, t['pos'][0]:t['pos'][1]] = 1.0 / len(tlist) / (t['pos'][1] - t['pos'][0])

                            relation_mask[i, j] = 1

                            delta_dis = hlist[0]['pos'][0] - tlist[0]['pos'][0]

                            if delta_dis < 0:
                                ht_pair_pos[i, j] = -int(self.dis2idx[-delta_dis])
                            else:
                                ht_pair_pos[i, j] = int(self.dis2idx[delta_dis])
                            j += 1

                max_h_t_cnt = max(max_h_t_cnt, j)
                label_set = {}
                for label in ins['labels']:
                    label_set[(label['h'], label['t'], label['r'])] = label['in'+self.train_prefix]

                labels.append(label_set)

                L_vertex.append(L)
                indexes.append(index)

            input_lengths = (context_idxs[:cur_bsz] > 0).long().sum(dim=1)
            max_c_len = int(input_lengths.max())

            entity_mention_num = list(map(add, entity_num, node_num[:cur_bsz].squeeze(1).tolist()))
            max_sdp_num = max(sdp_nums)
            all_node_num = list(map(add, sdp_nums , entity_mention_num))

            max_entity_num = max(entity_num)
            max_sentence_num = max(sentence_num)
            b_max_mention_num = int(node_num[:cur_bsz].max()) #- max_entity_num - max_sentence_num
            all_node_num = torch.LongTensor(all_node_num)

            yield {'context_idxs': context_idxs[:cur_bsz, :max_c_len].contiguous(),
                   'context_masks': context_masks[:cur_bsz, :max_c_len].contiguous(), #Ivan
                   'context_starts': context_starts[:cur_bsz, :max_c_len].contiguous(),
                   'context_pos': context_pos[:cur_bsz, :max_c_len].contiguous(),
                   'h_mapping': h_mapping[:cur_bsz, :max_h_t_cnt, :max_c_len],
                   't_mapping': t_mapping[:cur_bsz, :max_h_t_cnt, :max_c_len],
                   'context_seg': context_seg[:cur_bsz, :max_c_len].contiguous(),
                   'labels': labels,
                   'L_vertex': L_vertex,
                   'input_lengths': input_lengths,
                   'context_ner': context_ner[:cur_bsz, :max_c_len].contiguous(),
                   'context_char_idxs': context_char_idxs[:cur_bsz, :max_c_len].contiguous(),
                   'relation_mask': relation_mask[:cur_bsz, :max_h_t_cnt],
                   'titles': titles,
                   'ht_pair_pos': ht_pair_pos[:cur_bsz, :max_h_t_cnt],
                   'node_position': node_position[:cur_bsz, :b_max_mention_num, :max_c_len].contiguous(),
                   'node_sent_num': node_sent_num[:cur_bsz, :max_sentence_num].contiguous(),
                   'entity_position': entity_position[:cur_bsz, :max_entity_num, :max_c_len].contiguous(),
                   'indexes': indexes,
                   'all_node_num': all_node_num,
                   'entity_num': entity_num,
                   'sent_num': sentence_num,
                   'sdp_position': sdp_position[:cur_bsz, :max_sdp_num, :max_c_len].contiguous(),
                   'sdp_num': sdp_nums,
                   'vertexsets': vertexSets,
                   }

    def train(self, model_pattern, model_name):

        ori_model = model_pattern(config = self)
        if self.pretrain_model != None:
            ori_model.load_state_dict(torch.load(self.pretrain_model))
        ori_model.cuda()

        # parameters = [p for p in ori_model.parameters() if p.requires_grad]
        other_params = [p for name, p in ori_model.named_parameters() if p.requires_grad and not name.startswith("bert")]


        # optimizer = torch_utils.get_optimizer(self.optim, parameters, self.lr)
        optimizer = optim.Adam([
                        {"params": other_params, "lr":self.lr},
                        {"params": ori_model.bert.parameters(), "lr": 1e-5},
                        ], lr=self.lr)
        print(optimizer)

        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, self.lr_decay)

        model = nn.DataParallel(ori_model)

        BCE = nn.BCEWithLogitsLoss(reduction='none')

        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)

        best_auc = 0.0
        best_f1 = 0.0
        best_epoch = 0

        model.train()

        global_step = 0
        total_loss = 0
        start_time = time.time()

        def logging(s, print_=True, log_=True):
            if print_:
                print(s)
            if log_:
                with open(os.path.join(os.path.join("log", model_name)), 'a+') as f_log:
                    f_log.write(s + '\n')

        dev_score_list = []
        f1 = 0
        dev_score_list.append(f1)

        for epoch in range(self.max_epoch):
            gc.collect()
            self.acc_NA.clear()
            self.acc_not_NA.clear()
            self.acc_total.clear()
            print("epoch:{}, Learning rate:{}".format(epoch,optimizer.param_groups[0]['lr']))

            epoch_start_time = time.time()

            for no, data in enumerate(self.get_train_batch()):
                context_idxs = data['context_idxs']
                context_pos = data['context_pos']
                h_mapping = data['h_mapping']
                t_mapping = data['t_mapping']
                relation_label = data['relation_label']
                input_lengths =  data['input_lengths']
                relation_multi_label = data['relation_multi_label']
                relation_mask = data['relation_mask']
                context_ner = data['context_ner']
                context_char_idxs = data['context_char_idxs']
                ht_pair_pos = data['ht_pair_pos']
                context_seg = data['context_seg']
                context_masks = data['context_masks']
                context_starts = data['context_starts']

                dis_h_2_t = ht_pair_pos+10
                dis_t_2_h = -ht_pair_pos+10

                torch.cuda.empty_cache()

                context_idxs = context_idxs.cuda()
                context_pos = context_pos.cuda()
                context_ner = context_ner.cuda()
                #context_char_idxs = context_char_idxs.cuda()
                #input_lengths = input_lengths.cuda()
                h_mapping = h_mapping.cuda()
                t_mapping = t_mapping.cuda()
                relation_mask = relation_mask.cuda()
                dis_h_2_t = dis_h_2_t.cuda()
                dis_t_2_h = dis_t_2_h.cuda()

                node_position = data['node_position'].cuda()
                entity_position = data['entity_position'].cuda()
                node_sent_num = data['node_sent_num'].cuda()
                all_node_num = data['all_node_num'].cuda()
                entity_num = torch.Tensor(data['entity_num']).cuda()
                #sent_num = torch.Tensor(data['sent_num']).cuda()

                sdp_pos = data['sdp_position'].cuda()
                sdp_num = torch.Tensor(data['sdp_num']).cuda()

                predict_re = model(context_idxs, context_pos, context_ner,
                                   h_mapping, t_mapping, relation_mask, dis_h_2_t, dis_t_2_h, context_seg,
                                   node_position, entity_position, node_sent_num,
                                   all_node_num, entity_num, sdp_pos, sdp_num, context_masks, context_starts)

                relation_multi_label = relation_multi_label.cuda()

                loss = torch.sum(BCE(predict_re, relation_multi_label)*relation_mask.unsqueeze(2)) / torch.sum(relation_mask)

                output = torch.argmax(predict_re, dim=-1)
                output = output.data.cpu().numpy()

                optimizer.zero_grad()

                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
                optimizer.step()

                relation_label = relation_label.data.cpu().numpy()

                for i in range(output.shape[0]):
                    for j in range(output.shape[1]):
                        label = relation_label[i][j]
                        if label<0:
                            break
                        if label == 0:
                            self.acc_NA.add(output[i][j] == label)
                        else:
                            self.acc_not_NA.add(output[i][j] == label)

                        self.acc_total.add(output[i][j] == label)

                global_step += 1
                total_loss += loss.item()

                if global_step % self.period == 0:
                    cur_loss = total_loss / self.period
                    elapsed = time.time() - start_time
                    logging('| epoch {:2d} | step {:4d} |  ms/b {:5.2f} | train loss {:5.3f} | NA acc: {:4.2f} | not NA acc: {:4.2f}  | tot acc: {:4.2f} '.format(epoch, global_step, elapsed * 1000 / self.period, cur_loss, self.acc_NA.get(), self.acc_not_NA.get(), self.acc_total.get()))
                    total_loss = 0
                    start_time = time.time()

            if epoch > self.evaluate_epoch:

                logging('-' * 89)
                eval_start_time = time.time()
                model.eval()

                f1, f1_ig, auc, pr_x, pr_y = self.test(model, model_name)

                model.train()
                logging('| epoch {:3d} | time: {:5.2f}s'.format(epoch, time.time() - eval_start_time))
                logging('-' * 89)

                if f1 > best_f1:
                    best_f1 = f1
                    best_auc = auc
                    best_epoch = epoch
                    path = os.path.join(self.checkpoint_dir, model_name)
                    torch.save(ori_model.state_dict(), path)
                    logging("best f1 is: {}, epoch is: {}, save path is: {}".format(best_f1, best_epoch, path))

            if epoch > self.decay_epoch:  # and epoch < self.evaluate_epoch:# and epoch < self.evaluate_epoch:
                if self.optim == 'sgd' and f1 < dev_score_list[-1]:
                    self.lr *= self.lr_decay
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = self.lr

                if self.optim == 'adam' and optimizer.param_groups[0]['lr'] > 1e-4: #epoch < 30:# and f1 < dev_score_list[-1]:
                    scheduler.step()

            dev_score_list.append(f1)
            print("train time for epoch {}: {}".format(epoch,time.time()-epoch_start_time))

        print("Finish training")
        print("Best epoch = {} | F1 {}, auc = {}".format(best_epoch, best_f1, best_auc))
        print("Storing best result...")
        print("Finish storing")

    def test(self, model, model_name, output=False, input_theta=-1):
        data_idx = 0
        eval_start_time = time.time()
        test_result_ignore = []
        total_recall_ignore = 0

        test_result = []
        total_recall = 0
        top1_acc = have_label = 0

        def logging(s, print_=True, log_=True):
            if print_:
                print(s)
            if log_:
                with open(os.path.join(os.path.join("log", model_name)), 'a+') as f_log:
                    f_log.write(s + '\n')

        for i, data in enumerate(self.get_test_batch()):
            with torch.no_grad():
                context_idxs = data['context_idxs']
                context_pos = data['context_pos']
                h_mapping = data['h_mapping']
                t_mapping = data['t_mapping']
                labels = data['labels']
                L_vertex = data['L_vertex']
                #input_lengths =  data['input_lengths']
                context_ner = data['context_ner']
                #context_char_idxs = data['context_char_idxs']
                relation_mask = data['relation_mask']
                ht_pair_pos = data['ht_pair_pos']
                # Ivan
                context_masks = data['context_masks']
                context_starts = data['context_starts']

                titles = data['titles']
                indexes = data['indexes']

                context_seg = data['context_seg']

                dis_h_2_t = ht_pair_pos+10
                dis_t_2_h = -ht_pair_pos+10

                node_position = data['node_position'].cuda()
                entity_position = data['entity_position'].cuda()
                #node_position_sent = data['node_position_sent']#.cuda()
                node_sent_num = data['node_sent_num'].cuda()
                all_node_num = data['all_node_num'].cuda()
                entity_num = torch.Tensor(data['entity_num']).cuda()
                #sent_num = torch.Tensor(data['sent_num']).cuda()
                sdp_pos = data['sdp_position'].cuda()
                sdp_num = torch.Tensor(data['sdp_num']).cuda()
                predict_re = model(context_idxs, context_pos, context_ner,
                                   h_mapping, t_mapping, relation_mask, dis_h_2_t, dis_t_2_h, context_seg,
                                   node_position, entity_position, node_sent_num,
                                   all_node_num, entity_num, sdp_pos, sdp_num, context_masks, context_starts)

            predict_re = torch.sigmoid(predict_re)

            predict_re = predict_re.data.cpu().numpy()

            for i in range(len(labels)):
                label = labels[i]
                index = indexes[i]

                total_recall += len(label)
                for l in label.values():
                    if not l:
                        total_recall_ignore += 1

                L = L_vertex[i] # the number of entities in each instance.
                j = 0

                for h_idx in range(L):
                    for t_idx in range(L):
                       if h_idx != t_idx:

                            pre_r = np.argmax(predict_re[i, j])
                            if (h_idx, t_idx, pre_r) in label:
                                top1_acc += 1

                            flag = False

                            for r in range(1, self.relation_num):
                                intrain = False

                                if (h_idx, t_idx, r) in label:
                                    flag = True
                                    if label[(h_idx, t_idx, r)]==True:
                                        intrain = True
                                if not intrain:
                                    test_result_ignore.append( ((h_idx, t_idx, r) in label, float(predict_re[i,j,r]),  titles[i], self.id2rel[r], index, h_idx, t_idx, r, np.argmax(predict_re[i,j])))

                                test_result.append( ((h_idx, t_idx, r) in label, float(predict_re[i,j,r]),  titles[i], self.id2rel[r], index, h_idx, t_idx, r, pre_r) )
                            if flag:
                                have_label += 1

                            j += 1

            data_idx += 1

            if data_idx % self.period == 0:
                print('| step {:3d} | time: {:5.2f}'.format(data_idx // self.period, (time.time() - eval_start_time)))
                eval_start_time = time.time()

        test_result_ignore.sort(key=lambda x: x[1], reverse=True)
        test_result.sort(key = lambda x: x[1], reverse=True)

        print ('total_recall', total_recall)

        pr_x = []
        pr_y = []
        correct = 0
        w = 0

        if total_recall == 0:
            total_recall = 1  # for test

        for i, item in enumerate(test_result):
            correct += item[0]
            pr_y.append(float(correct) / (i + 1)) # precision
            pr_x.append(float(correct) / total_recall) # recall
            if item[1] > input_theta:
                w = i

        pr_x = np.asarray(pr_x, dtype='float32')
        pr_y = np.asarray(pr_y, dtype='float32')
        f1_arr = (2 * pr_x * pr_y / (pr_x + pr_y + 1e-20))
        f1 = f1_arr.max()
        f1_pos = f1_arr.argmax()
        theta = test_result[f1_pos][1]

        if input_theta==-1:
            w = f1_pos

        auc = sklearn.metrics.auc(x = pr_x, y = pr_y)
        if not self.is_test:
            logging('ALL   : Theta {:3.4f} | F1 {:3.4f} | Precision {:3.4f} | Recall {:3.4f} | AUC {:3.4f} '.format(theta, f1, pr_x[f1_pos], pr_y[f1_pos], auc))
        else:
            logging('ma_f1{:3.4f} | input_theta {:3.4f} test_result F1 {:3.4f} | AUC {:3.4f}'.format(f1, input_theta, f1_arr[w], pr_x[w], pr_y[w], auc))

        #logging("precision {}, recall {}".format(pr_y, pr_x))

        if output:
            # output = [x[-4:] for x in test_result[:w+1]]
            output = [{'index': int(x[-5]), 'h_idx': int(x[-4]), 't_idx': int(x[-3]), 'r_idx': int(x[-2]), 'r': x[-6], 'title': x[-7]} for x in test_result[:w+1]] # this is different from original paper
            json.dump(output, open(self.test_prefix + "_index.json", "w"))

        input_theta = theta
        pr_x = []
        pr_y = []
        correct = 0
        w = 0
        for i, item in enumerate(test_result_ignore):
            correct += item[0]
            pr_y.append(float(correct) / (i + 1))
            pr_x.append(float(correct) / total_recall)
            if item[1] > input_theta:
                w = i

        pr_x = np.asarray(pr_x, dtype='float32')
        pr_y = np.asarray(pr_y, dtype='float32')
        f1_arr = (2 * pr_x * pr_y / (pr_x + pr_y + 1e-20))
        f1_ig = f1_arr.max()

        auc = sklearn.metrics.auc(x = pr_x, y = pr_y)

        logging('Ignore ma_f1 {:3.4f} | input_theta {:3.4f} test_result F1 {:3.4f} | Precision {:3.4f}| Recall {:3.4f}| AUC {:3.4f}'.format(f1_ig, input_theta, f1_arr[w], pr_x[w], pr_y[w], auc))

        return f1, f1_ig, auc, pr_x, pr_y


    def testall(self, model_pattern, model_name, input_theta):#, ignore_input_theta):
        model = model_pattern(config = self)

        model.load_state_dict(torch.load(os.path.join(self.checkpoint_dir, model_name)))
        model.cuda()
        model.eval()
        f1, f1_ig, auc, pr_x, pr_y = self.test(model, model_name,  True, input_theta)


