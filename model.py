# coding=utf-8
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import nltk
import pickle
from config import config, logger, random_seed

torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)


class ParamContextModel(nn.Module):
    def __init__(self, total_ent_num, pem_tensor):
        super(ParamContextModel, self).__init__()
        # p(e|m)
        self.pem = nn.Parameter(pem_tensor)
        self.pem.requires_grad = False

        # WLM
        '''
        self.wlm = nn.Parameter(wlm_tensor)
        self.wlm.requires_grad = False
        '''
        self.wlm_scale = nn.Parameter(torch.ones(total_ent_num))

        self.ent_embeddings = nn.Embedding(total_ent_num, config.EMBEDDING_SIZE)
        self.ent_embeddings.weight.requires_grad = False
        self.linear_A = nn.Parameter(torch.ones(config.EMBEDDING_SIZE))
        self.linear_B = nn.Parameter(torch.ones(config.EMBEDDING_SIZE))
        self.linear_C = nn.Parameter(torch.ones(config.EMBEDDING_SIZE))
        self.threshold = nn.Threshold(0.0, -50)
        self.dropout = nn.Dropout(0.1)
        
        in_features = 4
        nn_pem_interm_size = config.EL_F_NETWORK_SIZE

        self.features_nums = 6
        if not os.path.exists('./mlp_layer.pkl'):
            self.mlp_layer = nn.Sequential(
                nn.Linear(self.features_nums, nn_pem_interm_size),
                nn.Dropout(0.1),
                nn.ReLU(),
                nn.Linear(nn_pem_interm_size, 1))
            print('mlp_layer.pkl not found, TRAIN FIRST!')
        else:
            with open('./mlp_layer.pkl', 'rb') as f:
                self.mlp_layer = pickle.load(f)
            print('mlp_layer.pkl found, LOAD SUCCESS!')
        
        self.f_network = nn.Sequential(
            nn.Linear(in_features, nn_pem_interm_size),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(nn_pem_interm_size, 1))
        
        # self.f_network = FNet()
        # self.f_network = nn.Linear(in_features, 1, bias=False)
        # self.f_network.weight.data = torch.ones(1, in_features)
        # self.f_network.weight.requires_grad = False
        

    def train_model(self):
        self.pem.requires_grad = False
        self.wlm_scale.requires_grad = False
        self.ent_embeddings.weight.requires_grad = False

        self.linear_A.requires_grad = True
        self.linear_B.requires_grad = True
        self.linear_C.requires_grad = True
        for p in self.f_network.parameters():
            p.requires_grad = True

    def train_feature(self):
        self.pem.requires_grad = True
        self.wlm_scale.requires_grad = True
        self.ent_embeddings.weight.requires_grad = True

        self.linear_A.requires_grad = False
        self.linear_B.requires_grad = False
        self.linear_C.requires_grad = False
        for p in self.f_network.parameters():
            p.requires_grad = False

    def generateFeature(self, data):
        candidate_ids, ctxt_word_vecs, mention_id, pem_candidate_ids, wlms, known_entities, known_entities_len = data
        pem = self.pem[mention_id]
        pem_index = torch.arange(pem_candidate_ids.size(0)).unsqueeze(-1).expand(pem_candidate_ids.size())
        pem = pem[pem_index, pem_candidate_ids]
        pem = F.normalize(pem, p=1, dim=1)

        candidate_vecs = self.ent_embeddings(candidate_ids)  # BS * CAND_NUM * VS
        candidate_vecs = F.normalize(candidate_vecs, dim=2)

        known_vecs = self.ent_embeddings(known_entities)
        known_vecs = F.normalize(known_vecs, dim=2)
        known_vecs = known_vecs.mean(dim=1)
        known_vecs = known_vecs.unsqueeze(1)
        coherence = torch.cosine_similarity(known_vecs, candidate_vecs, dim=2)  # BS * CAND_NUM

        ctxt_word_vecs = ctxt_word_vecs.mean(dim=1)
        ctxt_word_vecs = torch.unsqueeze(ctxt_word_vecs, 1)  # BS * 1 * VS
        context_sim = torch.cosine_similarity(ctxt_word_vecs, candidate_vecs, dim=2)  # BS * CAND_NUM

        context_sim = context_sim.unsqueeze(-1)
        coherence = coherence.unsqueeze(-1)
        pem = pem.unsqueeze(-1)

        feature = torch.cat([context_sim, pem, coherence], dim=-1)
        return feature

    def mlp(self, x, add_info=None, training=False):
        if not training:
            feat = self.generateFeature(x).cpu().detach().numpy()

            # calculate three features of strings (Yamada et al.)
            mention, candidates, id_names = add_info
            ment_surface = mention
            ment_surface = ment_surface.lower()
            cand_title = [id_names[id] for id in candidates]

            edit_dist = []
            eqaul_or_contain = []
            prefix_or_suffix = []
            for cand in cand_title:
                cand = cand.lower()
                edit_dist.append(nltk.edit_distance(ment_surface, cand))
                eqaul_or_contain.append(int(ment_surface in cand or ment_surface == cand))
                prefix_or_suffix.append(int(cand.startswith(ment_surface) or cand.endswith(ment_surface)))

            string_feats = np.array([edit_dist, eqaul_or_contain, prefix_or_suffix], dtype=feat.dtype)
            string_feats = string_feats.transpose()
            string_feats = np.expand_dims(string_feats, axis=0)
            string_feats = np.repeat(string_feats, feat.shape[0], axis=0)
            feat = np.concatenate((feat, string_feats), axis=-1)

            feat = torch.from_numpy(feat).float().to(device)

            logits = self.mlp_layer(feat).squeeze(-1)
        else:
            logits = self.mlp_layer(x).squeeze(-1)
        return logits
       

    def forward(self, x):
        # candidate_ids, ctxt_word_vecs, pes, wlms, known_entities, known_entities_len = x
        candidate_ids, ctxt_word_vecs, mention_id, pem_candidate_ids, wlms, known_entities, known_entities_len = x
        pem = self.pem[mention_id]
        pem_index = torch.arange(pem_candidate_ids.size(0)).unsqueeze(-1).expand(pem_candidate_ids.size())
        pem = pem[pem_index, pem_candidate_ids]
        pem = F.normalize(pem, p=1, dim=1)
        # print(pem)
        # exit(0)

        wlm_scales = self.wlm_scale[candidate_ids]
        wlms = wlms*wlm_scales

        ctxt_word_vecs = ctxt_word_vecs.transpose(1,2) # BS * VS * CTXT_NUM
        candidate_vecs = self.ent_embeddings(candidate_ids) # BS * CAND_NUM * VS
        candidate_vecs = F.normalize(candidate_vecs, dim=2)
        known_vecs = self.ent_embeddings(known_entities) # BS * KNOWN_NUM * VS
        known_vecs = F.normalize(known_vecs, dim=2)
        known_vecs = known_vecs.transpose(1,2) # BS * VS * KNOWN_NUM

        rel_vecs = candidate_vecs * self.linear_C
        #rel_vecs = self.dropout(rel_vecs)
        rel_sims = torch.bmm(rel_vecs, known_vecs) # BS * CAND_NUM * KNOWN_NUM
        rel_sims = rel_sims.sum(-1) # BS * CAND_NUM
        rel_sims = rel_sims.transpose(0, 1) / known_entities_len
        rel_sims = rel_sims.transpose(0, 1)

        att_vecs = candidate_vecs * self.linear_B
        #att_vecs = self.dropout(att_vecs)
        att_vecs = torch.bmm(att_vecs, ctxt_word_vecs) # BS * CAND_NUM * CTXT_NUM
        att_vecs, _ = torch.max(att_vecs, 1) # BS * CTXT_NUM
        att_topk, _ = att_vecs.topk(config.EL_R, 1) # BS * EL_R
        att_topk, _ = att_topk.min(1) # BS
        att_topk = att_topk.unsqueeze(-1) # BS * 1
        att_topk = att_topk.expand(-1, config.EL_CTXT_LEN) # BS * CTXT_NUM
        att_vecs = att_vecs - att_topk
        att_vecs = self.threshold(att_vecs)
        att_vecs = F.softmax(att_vecs, 1).unsqueeze(-1) # BS * CTXT_NUM * 1

        candidate_vecs = candidate_vecs * self.linear_A
        sims = torch.bmm(candidate_vecs, ctxt_word_vecs) # BS * CAND_NUM * CTXT_NUM
        # print(sims)
        scores = torch.bmm(sims, att_vecs) # BS * CAND_NUM * 1
        # scores = torch.sum(sims, -1)
        
        # scores = scores.unsqueeze(-1)
        pem = pem.unsqueeze(-1)
        wlms = wlms.unsqueeze(-1)
        rel_sims = rel_sims.unsqueeze(-1)
        # hiddens = torch.cat([scores, pes, wlms, rel_sims], dim=-1)
        # print('###')
        # print(pes[0])
        # print(wlms[0])
        hiddens = torch.cat([scores, pem, wlms, rel_sims], dim=-1)
        logits = self.f_network(hiddens).squeeze(-1)
        # logits =
        # return logits 
        # print(pem)
        return logits

# Deep Joint Entity Disambiguation with Local Neural Attention:
# Sec 4. {{Local Model with Neural Attention}}
class LocalEDModel(nn.Module):
    def __init__(self, config):
        pass

    def forward(self, x):
        pass

class FNet(nn.Module):
    def __init__(self):
        super(FNet, self).__init__()
        in_features = 4
        nn_pem_interm_size = config.EL_F_NETWORK_SIZE
        self.linear1 = nn.Linear(in_features, nn_pem_interm_size)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(nn_pem_interm_size)
        self.linear2 = nn.Linear(nn_pem_interm_size, 1)

    def forward(self, x):
        y = self.linear1(x)
        y = y.transpose(1,2)
        y = self.bn(y)
        y = y.transpose(1,2)
        y = self.relu(y)
        y = self.linear2(y)
        return y

class ContextModel(nn.Module):
    def __init__(self, total_ent_num):
        super(ContextModel, self).__init__()
        self.ent_embeddings = nn.Embedding(total_ent_num, config.EMBEDDING_SIZE)
        self.ent_embeddings.weight.requires_grad = False
        self.linear_A = nn.Parameter(torch.ones(config.EMBEDDING_SIZE))
        self.linear_B = nn.Parameter(torch.ones(config.EMBEDDING_SIZE))
        self.linear_C = nn.Parameter(torch.ones(config.EMBEDDING_SIZE))
        self.threshold = nn.Threshold(0.0, -50)
        self.dropout = nn.Dropout(0.1)
        
        in_features = 4
        nn_pem_interm_size = config.EL_F_NETWORK_SIZE
        
        self.f_network = nn.Sequential(
            nn.Linear(in_features, nn_pem_interm_size),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(nn_pem_interm_size, 1))
        
        # self.f_network = FNet()
        # self.f_network = nn.Linear(in_features, 1, bias=False)
        # self.f_network.weight.data = torch.ones(1, in_features)
        # self.f_network.weight.requires_grad = False
 

    def forward(self, x):
        candidate_ids, ctxt_word_vecs, pes, wlms, known_entities, known_entities_len = x
        '''
        ctxt_word_vecs = ctxt_word_vecs.transpose(1,2) # BS * VS * CTXT_NUM
        candidate_vecs = self.ent_embeddings(candidate_ids) # BS * CAND_NUM * VS
        known_vecs = self.ent_embeddings(known_entities) # BS * KNOWN_NUM * VS
        known_vecs = known_vecs.transpose(1,2) # BS * VS * KNOWN_NUM
        '''

        ctxt_word_vecs = ctxt_word_vecs.transpose(1,2) # BS * VS * CTXT_NUM
        candidate_vecs = self.ent_embeddings(candidate_ids) # BS * CAND_NUM * VS
        candidate_vecs = F.normalize(candidate_vecs, dim=2)
        known_vecs = self.ent_embeddings(known_entities) # BS * KNOWN_NUM * VS
        known_vecs = F.normalize(known_vecs, dim=2)
        known_vecs = known_vecs.transpose(1,2) # BS * VS * KNOWN_NUM

        rel_vecs = candidate_vecs * self.linear_C
        #rel_vecs = self.dropout(rel_vecs)
        rel_sims = torch.bmm(rel_vecs, known_vecs) # BS * CAND_NUM * KNOWN_NUM
        rel_sims = rel_sims.sum(-1) # BS * CAND_NUM
        rel_sims = rel_sims.transpose(0, 1) / known_entities_len
        rel_sims = rel_sims.transpose(0, 1)

        att_vecs = candidate_vecs * self.linear_B
        #att_vecs = self.dropout(att_vecs)
        att_vecs = torch.bmm(att_vecs, ctxt_word_vecs) # BS * CAND_NUM * CTXT_NUM
        att_vecs, _ = torch.max(att_vecs, 1) # BS * CTXT_NUM
        att_topk, _ = att_vecs.topk(config.EL_R, 1) # BS * EL_R
        att_topk, _ = att_topk.min(1) # BS
        att_topk = att_topk.unsqueeze(-1) # BS * 1
        att_topk = att_topk.expand(-1, config.EL_CTXT_LEN) # BS * CTXT_NUM
        att_vecs = att_vecs - att_topk
        att_vecs = self.threshold(att_vecs)
        att_vecs = F.softmax(att_vecs, 1).unsqueeze(-1) # BS * CTXT_NUM * 1

        candidate_vecs = candidate_vecs * self.linear_A
        sims = torch.bmm(candidate_vecs, ctxt_word_vecs) # BS * CAND_NUM * CTXT_NUM
        scores = torch.bmm(sims, att_vecs) # BS * CAND_NUM * 1
        # scores = torch.sum(sims, -1)
        
        # scores = scores.unsqueeze(-1)
        pes = pes.unsqueeze(-1)
        wlms = wlms.unsqueeze(-1)
        rel_sims = rel_sims.unsqueeze(-1)
        # hiddens = torch.cat([scores, pes, wlms, rel_sims], dim=-1)
        # print('###')
        # print(pes[0])
        # print(wlms[0])
        hiddens = torch.cat([scores, pes, wlms, rel_sims], dim=-1)
        logits = self.f_network(hiddens).squeeze()
        # print(logits[0])
        # logits =
        return logits 



# entity --> embedding
class EntToVecModel(nn.Module):
    def __init__(self, total_ent_num, pretrain_embeddings = None, embedding_size = 300):
        super(EntToVecModel, self).__init__()
        if pretrain_embeddings is not None:
            self.ent_embeddings = nn.Embedding.from_pretrained(pretrain_embeddings, freeze=False)
        else:
            self.ent_embeddings = nn.Embedding(total_ent_num, embedding_size)

    def forward(self, x):
        ctxt_word_vecs, ent_idxes = x
        ent_vecs = self.ent_embeddings(ent_idxes)
        ent_vecs = F.normalize(ent_vecs)
        ent_vecs = ent_vecs.view(config.BATCH_SIZE, -1, 1)
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
        self.golden_tokens = None
        self.token_mentions = None
        self.mentions = []
        self.known_mentions = []
        self.unknown_mentions = []
        self.is_anno = False

class Mention:
    def __init__(self):
        self.pos = []
        self.mention = None
        self.candidates = []
        self.golden_entity = None
        self.entity = None
        self.known = False
        self.entity_list = []
        self.score = None

def main():
    pass

if __name__ == '__main__':
    main()