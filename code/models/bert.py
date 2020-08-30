import torch
from torch import nn

#import pytorch_pretrained_bert as _bert
from pytorch_transformers import *

import numpy as np

#from dougu import flatten, lines



class Bert():

    MASK = '[MASK]'
    CLS = "[CLS]"
    SEP = "[SEP]"


    def __init__(self, model_class, model_name):
        super().__init__()
        self.model_name = model_name
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.model = model_class.from_pretrained(model_name).cuda()
        self.max_len = self.model.embeddings.position_embeddings.weight.size(0)
        self.dim = self.model.embeddings.position_embeddings.weight.size(1)

    def tokenize(self, text, masked_idxs=None):
        tokenized_text = self.tokenizer.tokenize(text)
        if masked_idxs is not None:
            for idx in masked_idxs:
                tokenized_text[idx] = self.MASK
        # prepend [CLS] and append [SEP]
        # see https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/examples/run_classifier.py#L195  # NOQA
        tokenized = [self.CLS] + tokenized_text + [self.SEP]
        return tokenized

    def tokenize_to_ids(self, text, masked_idxs=None, pad=True):
        tokens = self.tokenize(text, masked_idxs)
        return self.convert_tokens_to_ids(tokens, pad=pad)

    def convert_tokens_to_ids(self, tokens, pad=True):
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        ids = torch.tensor([token_ids])
        assert ids.size(1) < self.max_len
        if pad:
            padded_ids = torch.zeros(1, self.max_len).to(ids)
            padded_ids[0, :ids.size(1)] = ids
            mask = torch.zeros(1, self.max_len).to(ids)
            mask[0, :ids.size(1)] = 1
            return padded_ids, mask
        else:
            return ids

    def flatten(self, list_of_lists):
        for list in list_of_lists:
            for item in list:
                yield item

    def subword_tokenize(self, tokens):
        """Segment each token into subwords while keeping track of
        token boundaries.
        Parameters
        ----------
        tokens: A sequence of strings, representing input tokens.
        Returns
        -------
        A tuple consisting of:
            - A list of subwords, flanked by the special symbols required
                by Bert (CLS and SEP).
            - An array of indices into the list of subwords, indicating
                that the corresponding subword is the start of a new
                token. For example, [1, 3, 4, 7] means that the subwords
                1, 3, 4, 7 are token starts, while all other subwords
                (0, 2, 5, 6, 8...) are in or at the end of tokens.
                This list allows selecting Bert hidden states that
                represent tokens, which is necessary in sequence
                labeling.
        """
        subwords = list(map(self.tokenizer.tokenize, tokens))
        subword_lengths = list(map(len, subwords))
        subwords = [self.CLS] + list(self.flatten(subwords))[:509] + [self.SEP]
        token_start_idxs = 1 + np.cumsum([0] + subword_lengths[:-1])
        token_start_idxs[token_start_idxs > 509] = 509
        return subwords, token_start_idxs

    def subword_tokenize_to_ids(self, tokens):
        """Segment each token into subwords while keeping track of
        token boundaries and convert subwords into IDs.
        Parameters
        ----------
        tokens: A sequence of strings, representing input tokens.
        Returns
        -------
        A tuple consisting of:
            - A list of subword IDs, including IDs of the special
                symbols (CLS and SEP) required by Bert.
            - A mask indicating padding tokens.
            - An array of indices into the list of subwords. See
                doc of subword_tokenize.
        """
        subwords, token_start_idxs = self.subword_tokenize(tokens)
        subword_ids, mask = self.convert_tokens_to_ids(subwords)
        token_starts = torch.zeros(1, self.max_len).to(subword_ids)
        token_starts[0, token_start_idxs] = 1
        return subword_ids.numpy(), mask.numpy(), token_starts.numpy()

    def segment_ids(self, segment1_len, segment2_len):
        ids = [0] * segment1_len + [1] * segment2_len
        return torch.tensor([ids])
