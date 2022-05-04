# coding=utf-8
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import config

torch.manual_seed(666)
torch.cuda.manual_seed(666)
np.random.seed(666)
random.seed(666)


# Deep Joint Entity Disambiguation with Local Neural Attention:
# Sec 4. {{Local Model with Neural Attention}}




# 学习entity --> embedding
class EntToVecModel(nn.Module):
    def __init__(self, total_ent_num, pretrain_embeddings = None, embedding_size = 300):
        super(EntToVecModel, self).__init__()
        if pretrain_embeddings is not None:
            self.ent_embeddings = nn.Embedding.from_pretrained(pretrain_embeddings, freeze=False)
        else:
            self.ent_embeddings = nn.Embedding(total_ent_num, embedding_size)

    def forward(self, x):
        ctxt_word_vecs, ent_idxes = x
        ent_vecs = self.ent_embeddings(ent_idxes).view(config.BATCH_SIZE, -1, 1)
        ctxt_word_vecs = F.normalize(ctxt_word_vecs).view(config.BATCH_SIZE, config.NUM_WORDS_PER_ENT*config.NUM_NEG_WORDS, -1)
        sims = torch.matmul(ctxt_word_vecs, ent_vecs).view(config.BATCH_SIZE*config.NUM_WORDS_PER_ENT, -1)
        return sims


    def embedding(self, x):
        vecs = self.ent_embeddings(x)
        vecs = F.normalize(vecs)
        return vecs

class Query:
    def __init__(self):
        # self.init_wiki_id = None
        self.wiki_id = None
        self.mentions = []
        self.docs = []
        self.train_docs = []
        self.test_docs = []

    def train(self):
        train_query = Query()
        train_query.wiki_id = self.wiki_id
        train_query.mentions = self.mentions
        for idx in self.train_docs:
            train_query.docs.append(self.docs[idx])
        return train_query

    def test(self):
        test_query = Query()
        test_query.wiki_id = self.wiki_id
        test_query.mentions = self.mentions
        for idx in self.test_docs:
            test_query.docs.append(self.docs[idx])
        return test_query

class Doc:
    def __init__(self):
        self.id = None
        self.content = None
        self.tokens = None
        self.token_mentions = None
        self.mentions = []
        self.known_mentions = []
        self.unknown_mentions = []

class Mention:
    def __init__(self):
        self.pos = []
        self.mention = None
        self.candidates = []
        self.golden_entity = None
        self.entity = None
        self.known = False

def main():
    pass

if __name__ == '__main__':
    main()