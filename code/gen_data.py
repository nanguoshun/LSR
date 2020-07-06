import numpy as np
import os
import json
import argparse
import spacy
import networkx as nx

nlp = spacy.load("en_core_web_sm")

PAD_TOKEN = '<PAD>'
UNK_TOKEN = '<UNK>'

POS_TO_ID = {PAD_TOKEN: 0, UNK_TOKEN: 1, 'NNP': 2, 'NN': 3, 'IN': 4, 'DT': 5, ',': 6, 'JJ': 7, 'NNS': 8, 'VBD': 9, 'CD': 10, 'CC': 11, '.': 12, 'RB': 13, 'VBN': 14, 'PRP': 15, 'TO': 16, 'VB': 17, 'VBG': 18, 'VBZ': 19, 'PRP$': 20, ':': 21, 'POS': 22, '\'\'': 23, '``': 24, '-RRB-': 25, '-LRB-': 26, 'VBP': 27, 'MD': 28, 'NNPS': 29, 'WP': 30, 'WDT': 31, 'WRB': 32, 'RP': 33, 'JJR': 34, 'JJS': 35, '$': 36, 'FW': 37, 'RBR': 38, 'SYM': 39, 'EX': 40, 'RBS': 41, 'WP$': 42, 'PDT': 43, 'LS': 44, 'UH': 45, '#': 46}

MAX_SENT_LEN = 200
MAX_NODE_NUM = 200
MAX_ENTITY_NUM = 100
MAX_SENT_NUM = 30
MAX_NODE_PER_SENT = 40

parser = argparse.ArgumentParser()
parser.add_argument('--in_path', type = str, default =  "../data")
parser.add_argument('--out_path', type = str, default = "prepro_data")

args = parser.parse_args()
in_path = args.in_path
out_path = args.out_path
case_sensitive = False

char_limit = 16
train_distant_file_name = os.path.join(in_path, 'train_distant.json')
train_annotated_file_name = os.path.join(in_path, 'train_annotated.json')
dev_file_name = os.path.join(in_path, 'dev.json')
test_file_name = os.path.join(in_path, 'test.json')

rel2id = json.load(open(os.path.join(out_path, 'rel2id.json'), "r"))
id2rel = {v:u for u,v in rel2id.items()}
json.dump(id2rel, open(os.path.join(out_path, 'id2rel.json'), "w"))
fact_in_train = set([])
fact_in_dev_train = set([])

unk_number = 0

word_index_file_name = os.path.join(out_path,'vocab.pkl')

def GetNodePosition(data, node_position, node_position_sent, node_sent_num, entity_position, Ls):

    """
    :param data: input
    :param node_position: mention node position in a document (flatten)
    :param node_position_sent: node position in each sentence of a document
    :param node_sent_num: number of nodes in each sentence
    :param entity_position:
    :param Ls: the start position of each sentence in document
    :return:
    """
    nodes = [[] for _ in range(len(data['sents']))]
    nodes_sent = [[] for _ in range(len(data['sents']))]

    for ns_no, ns in enumerate(data['vertexSet']):
        for n in ns:
            sent_id = int(n['sent_id'])
            doc_pos_s =  n['pos'][0] + Ls[sent_id]
            doc_pos_e =  n['pos'][1] + Ls[sent_id]
            assert( doc_pos_e <= Ls[-1] )
            nodes[sent_id].append([sent_id]+[ns_no] + [doc_pos_s, doc_pos_e])
            nodes_sent[sent_id].append([sent_id] + n['pos'])
    id = 0

    for ns in nodes:
        for n in ns:
            n.insert(0, id)
            id += 1

    assert (id <= MAX_NODE_NUM)

    entity_num = len(data['vertexSet'])
    #sent_num = len(data['sents'])

    # generate entities(nodes) mask for document
    for ns in nodes:
        for n in ns:
            node_position[n[0]][n[3]:n[4]] = 1

    # generate entities(nodes) mask for sentences in a document
    for sent_no, ns in enumerate(nodes_sent):
        #print("len of ns is {}".format(len(ns)))
        assert ( len(ns) < MAX_NODE_PER_SENT )
        node_sent_num[sent_no] = len(ns)
        for n_no, n in enumerate(ns): # node no in a sentence
            assert (sent_no == n[0])
            node_position_sent[sent_no][n_no][n[1]:n[2]] = 1

    # entity matrixs
    for e_no, es in enumerate(data['vertexSet']):
        for e in es:
            sent_id = int(e['sent_id'])
            doc_pos_s =  e['pos'][0] + Ls[sent_id]
            doc_pos_e =  e['pos'][1] + Ls[sent_id]
            entity_position[e_no][doc_pos_s:doc_pos_e] = 1

    total_mentions = id #+ entity_num + sent_num

    total_num_nodes = total_mentions + entity_num
    assert (total_num_nodes <= MAX_NODE_NUM)

    return  total_mentions #only mentions

def ExtractMDPNode(data, sdp_pos, sdp_num, Ls):
    """
    Extract MDP node for each document
    :param data:
    :param sdp_pos: sdp here indicates shortest dependency path:
    :return:
    """
    sents = data["sents"]
    nodes = [[] for _ in range(len(data['sents']))]
    sdp_lists = []
    #create mention's list for each sentence
    for ns_no, ns in enumerate(data['vertexSet']):
        for n in ns:
            sent_id = int(n['sent_id'])
            nodes[sent_id].append(n['pos'])

    for sent_no in range(len(sents)):
        spacy_sent = nlp(' '.join(sents[sent_no]))
        edges = []
        if len(spacy_sent) != len(sents[sent_no]):
            #print("{}th doc {}th sent. not valid spacy parsing as the length is not the same to original sentence. ".format(doc_no, sent_no))
            sdp_lists.append([])
            continue
#       assert (len(spacy_sent) == len(sents[sent_no])) # make sure the length of sentence parsed by spacy is the same as original sentence.
        for token in spacy_sent:
            for child in token.children:
                edges.append(('{0}'.format(token.i),'{0}'.format(child.i)))

        graph = nx.Graph(edges)# Get the length and path

        mention_num = len(nodes[sent_no])
        sdp_list = []
        # get the shortest dependency path of all mentions in a sentence
        entity_indices  = []
        for m_i in range(mention_num): # m_i is the mention number
            indices_i = [nodes[sent_no][m_i][0] + offset for offset in range(nodes[sent_no][m_i][1] - nodes[sent_no][m_i][0])]
            entity_indices = entity_indices + indices_i
            for m_j in range(mention_num): #
                if m_i == m_j:
                    continue
                indices_j = [nodes[sent_no][m_j][0] + offset for offset in range(nodes[sent_no][m_j][1] - nodes[sent_no][m_j][0])]
                for index_i in indices_i:
                    for index_j in indices_j:
                        try:
                            sdp_path = nx.shortest_path(graph, source='{0}'.format(index_i), target='{0}'.format(index_j))
                        except (nx.NetworkXNoPath,nx.NodeNotFound) as e:
                            #print("no path")
                            #print(e)
                            continue
                        sdp_list.append(sdp_path)
        #get the sdp indices in a sentence
        sdp_nodes_flat = [sdp for sub_sdp in sdp_list for sdp in sub_sdp]
        entity_set = set(entity_indices)
        sdp_nodes_set = set(sdp_nodes_flat)
        #minus the entity node
        sdp_list = list(set([int(n) for n in sdp_nodes_set]) - entity_set)
        sdp_list.sort()
        sdp_lists.append(sdp_list)

    # calculate the sdp position in a document
    if len(sents) != len(sdp_lists):
        print("len mismatch")
    for i in range(len(sents)):
        if len(sdp_lists[i]) == 0:
            continue
        for j, sdp in enumerate(sdp_lists[i]):
            if j > len(sdp_lists[i]) - 1:
                print("list index out of range")
            sdp_lists[i][j] = sdp + Ls[i]

    flat_sdp = [sdp for sub_sdp in sdp_lists for sdp in sub_sdp]

    # set the sdp poistion as 1. for example, if the sdp_pos size is 100 X 512, then we will set the value in each row as 1 according to flat_sdp[i]
    for i in range(len(flat_sdp)):
        if i > MAX_ENTITY_NUM-1:
            continue
        sdp_pos[i][flat_sdp[i]] = 1

    sdp_num[0] = len(flat_sdp)

def Init(data_file_name, rel2id, max_length = 512, is_training = True, suffix=''):

    ori_data = json.load(open(data_file_name))
    sen_tot = len(ori_data)

    Ma = 0
    Ma_e = 0
    data = []
    intrain = notintrain = notindevtrain = indevtrain = 0

    node_position = np.zeros((sen_tot, MAX_NODE_NUM, max_length), dtype = np.int16)
    node_position_sent = np.zeros((sen_tot, MAX_SENT_NUM, MAX_NODE_PER_SENT, MAX_SENT_LEN), dtype = np.int16)
    node_sent_num = np.zeros((sen_tot, MAX_SENT_NUM), dtype = np.int16)
    entity_position = np.zeros((sen_tot, MAX_ENTITY_NUM, max_length), dtype= np.int16)
    node_num = np.zeros((sen_tot, 1), dtype= np.int16)

    sdp_position = np.zeros((sen_tot, MAX_ENTITY_NUM, max_length), dtype= np.int16)
    sdp_num = np.zeros((sen_tot, 1),dtype= np.int16)

    for i in range(len(ori_data)):
        if i % 200 == 0:
            print("generating the {}th instance from the file {}".format(i,data_file_name))
        Ls = [0]
        L = 0
        for x in ori_data[i]['sents']:
            L += len(x)
            Ls.append(L)

        node_num[i] = GetNodePosition(ori_data[i], node_position[i], node_position_sent[i], node_sent_num[i], entity_position[i], Ls)

        ExtractMDPNode(ori_data[i], sdp_position[i], sdp_num[i], Ls)

        vertexSet =  ori_data[i]['vertexSet']
        # point position added with sent start position
        for j in range(len(vertexSet)):
            for k in range(len(vertexSet[j])):
                vertexSet[j][k]['sent_id'] = int(vertexSet[j][k]['sent_id'])

                sent_id = vertexSet[j][k]['sent_id']
                dl = Ls[sent_id]
                pos1 = vertexSet[j][k]['pos'][0]
                pos2 = vertexSet[j][k]['pos'][1]
                vertexSet[j][k]['pos'] = (pos1+dl, pos2+dl)

        ori_data[i]['vertexSet'] = vertexSet

        item = {}
        item['vertexSet'] = vertexSet
        labels = ori_data[i].get('labels', [])

        train_triple = set([])
        new_labels = []
        for label in labels:
            rel = label['r']
            assert(rel in rel2id)
            label['r'] = rel2id[label['r']]

            train_triple.add((label['h'], label['t']))

            if suffix=='_train':
                for n1 in vertexSet[label['h']]:
                    for n2 in vertexSet[label['t']]:
                        fact_in_dev_train.add((n1['name'], n2['name'], rel))

            if is_training:
                for n1 in vertexSet[label['h']]:
                    for n2 in vertexSet[label['t']]:
                        fact_in_train.add((n1['name'], n2['name'], rel))

            else:
                for n1 in vertexSet[label['h']]:
                    for n2 in vertexSet[label['t']]:
                        if (n1['name'], n2['name'], rel) in fact_in_train:
                            intrain += 1
                            label['intrain'] = True
                        else:
                            notintrain += 1
                            label['intrain'] = False

                        if suffix == '_dev' or suffix == '_test':
                            if (n1['name'], n2['name'], rel) in fact_in_dev_train:
                                indevtrain += 1
                                label['indev_train'] = True
                            else:
                                notindevtrain += 1
                                label['indev_train'] = False

            new_labels.append(label)

        item['labels'] = new_labels
        item['title'] = ori_data[i]['title']

        na_triple = []
        for j in range(len(vertexSet)):
            for k in range(len(vertexSet)):
                if (j != k):
                    if (j, k) not in train_triple:
                        na_triple.append((j, k))

        item['na_triple'] = na_triple
        item['Ls'] = Ls
        item['sents'] = ori_data[i]['sents']
        data.append(item)

        Ma = max(Ma, len(vertexSet))
        Ma_e = max(Ma_e, len(item['labels']))


    print ('data_len:', len(ori_data))

    print ('fact_in_train', len(fact_in_train))
    print (intrain, notintrain)
    print ('fact_in_devtrain', len(fact_in_dev_train))
    print (indevtrain, notindevtrain)

    # saving
    print("Saving files")
    if is_training:
        name_prefix = "train"
    else:
        name_prefix = "dev"

    json.dump(data , open(os.path.join(out_path, name_prefix + suffix + '.json'), "w"))

    char2id = json.load(open(os.path.join(out_path, "char2id.json")))
    word2id = json.load(open(os.path.join(out_path, "word2id.json")))

    ner2id = json.load(open(os.path.join(out_path, "ner2id.json")))

    sen_word = np.zeros((sen_tot, max_length), dtype = np.int64)
    sen_wordstr = np.zeros((sen_tot, max_length), dtype = np.object)
    sen_pos = np.zeros((sen_tot, max_length), dtype = np.int16)
    sen_ner = np.zeros((sen_tot, max_length), dtype = np.int16)
    sen_char = np.zeros((sen_tot, max_length, char_limit), dtype = np.int16)
    sen_seg = np.zeros((sen_tot, max_length), dtype= np.int16)

    unkown_words = set()

    for i in range(len(ori_data)):

        item = ori_data[i]
        words = []
        sen_seg[i][0] = 1
        for sent in item['sents']:
            words += sent
            sen_seg[i][len(words)-1] = 1

        for j, word in enumerate(words):
            word = word.lower()
            sen_wordstr[i][j] = word
            #print(sen_wordstr[i][j])
            if j < max_length:
                if word in word2id:
                    sen_word[i][j] = word2id[word]
                else:
                    sen_word[i][j] = word2id['UNK']
                    unkown_words.add(word)
                if sen_word[i][j] < 0:
                    print("the id should not be negative")
            for c_idx, k in enumerate(list(word)):
                if c_idx>=char_limit:
                    break
                sen_char[i,j,c_idx] = char2id.get(k, char2id['UNK'])

        for j in range(j + 1, max_length):
                sen_word[i][j] = word2id['BLANK']

        vertexSet = item['vertexSet']

        for idx, vertex in enumerate(vertexSet, 1):
            for v in vertex:
                sen_pos[i][v['pos'][0]:v['pos'][1]] = idx
                ner_type_B = ner2id[v['type']]
                ner_type_I = ner_type_B+1
                sen_ner[i][v['pos'][0]] = ner_type_B
                sen_ner[i][v['pos'][0] + 1:v['pos'][1]] = ner_type_I

    print("Finishing processing")

    np.save(os.path.join(out_path, name_prefix + suffix + '_word.npy'), sen_word)
    np.save(os.path.join(out_path, name_prefix + suffix + '_pos.npy'), sen_pos)
    np.save(os.path.join(out_path, name_prefix + suffix + '_ner.npy'), sen_ner)
    np.save(os.path.join(out_path, name_prefix + suffix + '_char.npy'), sen_char)
    np.save(os.path.join(out_path, name_prefix + suffix + '_wordstr.npy'), sen_wordstr)
    np.save(os.path.join(out_path, name_prefix + suffix + '_seg.npy'), sen_seg)
    np.save(os.path.join(out_path, name_prefix + suffix + '_node_position.npy'), node_position)
    np.save(os.path.join(out_path, name_prefix + suffix + '_node_position_sent.npy'), node_position_sent)
    np.save(os.path.join(out_path, name_prefix + suffix + '_node_num.npy'), node_num)
    np.save(os.path.join(out_path, name_prefix + suffix + '_entity_position.npy'), entity_position)
    np.save(os.path.join(out_path, name_prefix + suffix + '_sdp_position.npy'), sdp_position)
    np.save(os.path.join(out_path, name_prefix + suffix + '_sdp_num.npy'), sdp_num)
    np.save(os.path.join(out_path, name_prefix + suffix + '_node_sent_num.npy'), node_sent_num)

    print("unk number for {} is: {}".format(suffix, len(unkown_words)))
    print("Finish saving")

print("=========================start to generate the training instances=========================")
Init(train_annotated_file_name, rel2id, max_length = 512, is_training = False, suffix='_train')
print("=========================start to generate the dev instances=========================")
Init(dev_file_name, rel2id, max_length = 512, is_training = False, suffix='_dev')
print("=========================start to generate the test instances=========================")
Init(test_file_name, rel2id, max_length = 512, is_training = False, suffix='_test')


