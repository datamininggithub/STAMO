from base_linker import Linker, get_origin_entity, load_wiki_df, load_wiki_wlm, load_wiki_dict
import pickle, random, math, time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from config import config, logger, device, random_seed
from utils import *
from model import ParamContextModel
from collections import OrderedDict, defaultdict
from feature import wlm_feature, pem_feature, embedding_feature, is_stop_word_or_number
from tqdm import tqdm

torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)

def mention_distance(m1, m2):
    min_dist = 9999999999
    for p1 in m1.pos:
        for p2 in m2.pos:
            if p1[0] >= p2[1]:
                dist = p1[0] - p2[1]
            else:
                dist = p2[0] - p1[1]
            if dist < 0:
                print(m1.pos, m2.pos)
            assert dist >=0
            min_dist = min(min_dist, dist)
    return min_dist

# dict: key --> (postive) int
def normalize_dict(int_dict):
    max_v = -1
    norm_dict = {}
    for k, v in int_dict.items():
        max_v = max(max_v, math.log(v+1))
    for k, v in int_dict.items():
        norm_dict[k] = math.log(v+1) / max_v
    return norm_dict

def wlm_to_tensor(wlm, ent_utils, ne_to_id):
    t = torch.zeros(len(ne_to_id), ent_utils.get_total_ent_num()).to(device)
    for wiki_id, counts in wlm.items():
        if wiki_id not in ne_to_id:
            continue
        idx = ne_to_id[wiki_id]
        for other_wiki_id, score in counts.items():
            other_idx = ent_utils.get_id_from_ent(other_wiki_id)
            t[idx][other_idx] = score
    return t

class ParamContextLinker(Linker):
    def __init__(self, names, candidates, saved_model_path = None, saved_full_model = False):
        super().__init__(names)
        # self.names = names
        # self.candidates = candidates

        self.word_utils = WordUtils()
        self.wiki_df = load_wiki_df()
        self.wiki_dict = load_wiki_dict()
        self.wiki_wlm = load_wiki_wlm()
        # self.wiki_wlm = defaultdict(dict)

        logger.info('load word utils.')
        ent_utils_path = config.ent_utils_path()
        with open(ent_utils_path, 'rb') as fin:
            data = pickle.load(fin)
            ent_utils = data['ent_utils']
            self.ent_utils = ent_utils
            self.ent_init_embeddings = data['ent_init_embeddings']
            logger.info('load entity utils.')
        ent_model_path = config.ent_model_path()
        pretrained_state_dict = torch.load(ent_model_path, map_location=device)
        # pretrained_state_dict = OrderedDict({k: v.float() for k, v in pretrained_state_dict.items()})
        ent_param = pretrained_state_dict['ent_embeddings.weight']
        ent_param[0] = torch.zeros(config.EMBEDDING_SIZE)

        self.new_entity_num = len(names)
        self.ne_to_id = {}
        # mention -> id
        mention_to_id = {}
        for wiki_id, mentions in names.items():
            self.ne_to_id[wiki_id] = len(self.ne_to_id)
            for mention in mentions:
                if mention not in mention_to_id:
                    mention_to_id[mention] = len(mention_to_id)
        self.mention_to_id = mention_to_id

        # re-map candidate -> id
        candidate_to_id = {}
        candidates = sorted(candidates)
        for candidate in candidates:
            if candidate in names:
                continue
            assert candidate not in candidate_to_id
            candidate_to_id[candidate] = len(candidate_to_id)
        for wiki_id in names:
            assert wiki_id not in candidate_to_id
            candidate_to_id[wiki_id] = len(candidate_to_id)
        self.candidate_to_id = candidate_to_id

        # tensor for p(e|m)
        pem_tensor = torch.zeros(len(mention_to_id), len(candidate_to_id))
        for mention, mid in mention_to_id.items():
            pem = torch.zeros(len(candidate_to_id))
            counts = self.wiki_dict[mention] if mention in self.wiki_dict else {}
            '''
            pes = [counts[x] if (x != 0 and x in counts) else 0 for x in candidates]
            pes = [float(pe) for pe in pes]
            pes = torch.tensor(pes)
            pes = F.normalize(pes, p=1, dim=0)
            '''
            for candidate, count in counts.items():
                if candidate not in candidate_to_id:
                    continue
                pem[candidate_to_id[candidate]] = count
            pem = F.normalize(pem, p=1, dim=0)
            pem_tensor[mid] = pem

        # tensor for p(e|m)
        '''
        wlm_tensor = torch.zeros(self.new_entity_num, ent_utils.get_total_ent_num())
        for i, wiki_id in enumerate(names):
            assert wiki_id in self.wiki_wlm
            counts = self.wiki_wlm[wiki_id]
            wlm = torch.zeros(ent_utils.get_total_ent_num())
            for other_wiki_id, score in counts.items():
                wlm[ent_utils.get_id_from_ent(other_wiki_id)] = score
            wlm_tensor[i] = wlm
        '''

        self.data = []
        model = ParamContextModel(ent_utils.get_total_ent_num(), pem_tensor)
        model_state_dict = model.state_dict()
        model.load_state_dict(pretrained_state_dict, strict=False)
        parameters = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = torch.optim.SGD(parameters, lr=config.EL_LEARNING_RATE)
        criterion = nn.MultiMarginLoss(margin=0.01)

        embedding_data = model.ent_embeddings.weight.data
        # wlm_data = model.wlm.data
        start_idx = ent_utils.get_total_ent_num() - self.new_entity_num
        for i, source_id in enumerate(names):
            source_idx = ent_utils.get_id_from_ent(source_id)
            target_idx = start_idx + i
            target_id = ent_utils.get_ent_from_id(target_idx)
            ent_utils.ent_to_idx[source_id] = target_idx
            ent_utils.ent_to_idx[target_id] = source_idx
            ent_utils.idx_to_ent[source_idx] = target_id
            ent_utils.idx_to_ent[target_idx] = source_id
            tmp = embedding_data[source_idx].clone().detach()
            embedding_data[source_idx] = embedding_data[target_idx]
            embedding_data[target_idx] = tmp


        if saved_model_path:
            if saved_full_model:
                saved_model = torch.load(saved_model_path, map_location=device)
                model.load_state_dict(saved_model['state_dict'])
                self.wiki_wlm = saved_model['wiki_wlm']
                self.candidate_to_id = saved_model['candidate_to_id']
            else:
                saved_model = torch.load(saved_model_path, map_location=device)
                del saved_model['ent_embeddings.weight']
                model.load_state_dict(saved_model, strict = False)
        self.is_train_feature = False

        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        # self.normalize_f_net()

        self.step = 0
        '''
        [model.pem.data - last_pem, 
                cur_wlm_tensor - last_wlm_tensor,
                model.wlm_scale.data - last_wlm_scale,
                model.ent_embeddings.weight.data - last_ent_embeddings]
        '''
        self.ms = [torch.zeros(model.pem.data.size()).to(device),
                torch.zeros(len(self.ne_to_id), ent_utils.get_total_ent_num()).to(device),
                torch.zeros(model.wlm_scale.data.size()).to(device),
                torch.zeros(len(self.ne_to_id), model.ent_embeddings.weight.data.size(1)).to(device)
            ]
        self.vs=  [torch.zeros(model.pem.data.size()).to(device),
                torch.zeros(len(self.ne_to_id), ent_utils.get_total_ent_num()).to(device),
                torch.zeros(model.wlm_scale.data.size()).to(device),
                torch.zeros(len(self.ne_to_id), model.ent_embeddings.weight.data.size(1)).to(device)
            ]

    def train_model(self):
        self.is_train_feature = False
        model = self.model
        model.train_model()
        parameters = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = torch.optim.SGD(parameters, lr=config.EL_LEARNING_RATE)
        self.optimizer = optimizer

    def train_feature(self):
        self.is_train_feature = True
        model = self.model
        model.train_feature()
        parameters = filter(lambda p: p.requires_grad, model.parameters())
        # optimizer = torch.optim.SGD(parameters, lr=config.EL_LEARNING_RATE)
        param_group = [{'params': model.pem}, {'params': model.wlm_scale, 'lr':config.EL_LEARNING_RATE*10000}, {'params': model.ent_embeddings.weight, 'lr':config.EL_LEARNING_RATE*10000}]
        optimizer = torch.optim.SGD(param_group, lr=config.EL_LEARNING_RATE)
        self.optimizer = optimizer

    def inter_slot_opt(self, lr = 0.05, beta1=0.9, beta2=0.999, wds=1e-3):
        default_lr = 5e-2
        last_wiki_wlm = self.last_param[0]
        last_pem = self.last_param[1]
        last_wlm_scale = self.last_param[2]
        last_ent_embeddings = self.last_param[3]

        ent_utils = self.ent_utils
        ne_to_id = self.ne_to_id
        model = self.model
        # s_{t-1}
        ms = self.ms
        # v_{t-1}
        vs = self.vs
        last_wlm_tensor = wlm_to_tensor(last_wiki_wlm, ent_utils, ne_to_id)
        cur_wlm_tensor = wlm_to_tensor(self.wiki_wlm, ent_utils, ne_to_id)

        # -\delta^t
        gs = [model.pem.data - last_pem, 
                cur_wlm_tensor - last_wlm_tensor,
                model.wlm_scale.data - last_wlm_scale,
                model.ent_embeddings.weight.data[-len(ne_to_id):,:] - last_ent_embeddings[-len(ne_to_id):,:]]

        self.step += 1
        # WARM-UP:
        lr = min(default_lr, default_lr * self.step / 5)
        for i in range(4):
            gs[i] = gs[i].to(device)
            
            ms[i] = beta1 * ms[i] + (1-beta1) * gs[i]
            vs[i] = beta2 * vs[i] + (1-beta2) * (gs[i]**2)
            m = ms[i] / (1 - beta1 ** self.step)
            v = vs[i] / (1 - beta2 ** self.step)
            # \hat{\delta}^t
            gs[i] = m / (v**0.5+1e-8)
            

        
        model.pem.data = last_pem + lr * gs[0]
        cur_wlm_tensor = last_wlm_tensor + lr * gs[1]
        model.wlm_scale.data = last_wlm_scale + lr * gs[2]
        model.ent_embeddings.weight.data[-len(ne_to_id):,:] = last_ent_embeddings[-len(ne_to_id):,:] + (lr/50) * gs[3]

        for wiki_id, nid in ne_to_id.items():
            wlm = {}
            t = cur_wlm_tensor[nid]
            # mask_t = (t>0)*t
            indices = torch.nonzero(t).squeeze().tolist()
            for indice in indices:
                other_wiki_id = ent_utils.get_ent_from_id(indice)
                wlm[other_wiki_id] = t[indice]
            self.wiki_wlm[wiki_id] = wlm

    def update_feature(self, dataset, vocab, names, samples, valid_func=None, max_epoch=600, lr=5e-2):
        logger.info('normal update feature.')
        model = self.model
        last_wiki_wlm = {}
        for k, v in self.wiki_wlm.items():
            last_wiki_wlm[k] = {}
            for kk, cnt in v.items():
                last_wiki_wlm[k][kk] = cnt
        last_pem = model.pem.data.clone().detach() 
        last_wlm_scale = model.wlm_scale.data.clone().detach() 
        last_ent_embeddings = model.ent_embeddings.weight.data.clone().detach() 
        self.last_param = [last_wiki_wlm, last_pem, last_wlm_scale, last_ent_embeddings]

        # update wlm
        logger.info('updating wlm feature...')
        f = wlm_feature(dataset, samples)
        for k, v in f.items():
            self.wiki_wlm[k] = v
        
        
        # update p(e|m)
        logger.info('updating p(e|m) feature...')
        f = pem_feature(dataset)
        candidate_to_id = self.candidate_to_id
        mention_to_id = self.mention_to_id
        new_entity_num = self.new_entity_num
        pem_tensor = self.model.pem.data
        pem_tensor[:,-self.new_entity_num:] = 0
        pem_tensor = F.normalize(pem_tensor, p=1, dim=1)

        for wiki_id in dataset:
            cid = candidate_to_id[wiki_id]
            mentions = names[wiki_id]
            for mention in mentions:
                mid = mention_to_id[mention]
                # pem_tensor[mid][cid] = 0.0
                # pem_tensor[mid] = F.normalize(pem_tensor[mid], p=1, dim=0)
                if mention not in f or wiki_id not in f[mention]:
                    continue
                if pem_tensor[mid].sum().item() == 0:
                    continue
                x = 0.0
                y = 0.0
                for entity in f[mention]:
                    if entity == wiki_id:
                        x += f[mention][entity]
                    y += f[mention][entity]
                p = x / y if y > 0 else 0.0
                if p > 0  and p < 1:
                    pem_tensor[mid][cid] = p / (1-p)
                    # pem_tensor[mid] = F.normalize(pem_tensor[mid], p=1, dim=0)
        self.model.pem.data = pem_tensor

        logger.info('test before update:')
        if valid_func is not None:
            valid_func(linker=self)
        # update embedding
        init_embeddings = None
        init_embeddings = {}
        for wiki_id in dataset:
            init_embeddings[wiki_id] = self.ent_init_embeddings[self.ent_utils.get_id_from_ent(wiki_id)]

        logger.info('updating embeddings...')
        wid_to_word = {v:k for k, v in vocab.items()}
        model = self.model
        embedding_data = model.ent_embeddings.weight.data
        ent_utils = self.ent_utils
        for wiki_id in dataset:
            # embedding_data[ent_utils.get_id_from_ent(wiki_id)] = torch.zeros(config.EMBEDDING_SIZE)
            embedding_data[ent_utils.get_id_from_ent(wiki_id)] = torch.tensor(self.ent_init_embeddings[self.ent_utils.get_id_from_ent(wiki_id)])
        
        
        for epoch, f in embedding_feature(dataset, wid_to_word, names, init_embeddings=init_embeddings, max_epoch = max_epoch, lr=lr):
            for k, v in f.items():
                embedding_data[ent_utils.get_id_from_ent(k)] = v
            if epoch > 200:
                if valid_func is not None and epoch % 10 == 0:
                    valid_func(linker=self)
                    pass
        


    def save(self, name):
        model_path = './model/linking_model.{}.{}.ckpt'.format(config.TASK, name)
        model = self.model
        param = {'state_dict': model.state_dict(), 'wiki_wlm': self.wiki_wlm, 'candidate_to_id': self.candidate_to_id}
        # torch.save(model.state_dict(), model_path)
        torch.save(param, model_path)
        logger.info("save model to: {}.".format(model_path))


    def add_train_instance(self, mention, doc, vocab):
        word_utils = self.word_utils

        # get context
        unknown_mention = mention
        pos = mention.pos
        mention_to_token = doc.mention_to_token
        pos = [mention_to_token[p] for p in pos if p in mention_to_token]
        assert len(pos) > 0
        cur_pos = random.choice(pos)


        tokens = doc.tokens
        half_ctxt_len = config.EL_CTXT_LEN // 2
        ctxt = []
        left_ctxt = tokens[max(0, cur_pos[0]-half_ctxt_len) :cur_pos[0]]
        ctxt += left_ctxt
        right_ctxt = tokens[cur_pos[1]: min(cur_pos[1]+half_ctxt_len, len(tokens))]
        ctxt += right_ctxt
        ctxt = [vocab[wid] for wid in ctxt]
        ctxt_ids = []
        for word in ctxt:
            if word_utils.contains_word(word):
                ctxt_ids.append(word_utils.get_id_from_word(word))
        if len(ctxt_ids) == 0:
            return
        assert len(ctxt_ids) > 0 and len(ctxt_ids) <= config.EL_CTXT_LEN
        ctxt_ids += [0] * (config.EL_CTXT_LEN - len(ctxt_ids))
        ctxt_word_vecs = word_utils.get_embeddings(torch.tensor(ctxt_ids))

        # get candidate entities
        wiki_df = self.wiki_df
        ent_utils = self.ent_utils
        candidates = [get_origin_entity(candidate) for candidate in mention.candidates]
        golden_entity = get_origin_entity(mention.golden_entity)
        candidates = sorted(candidates, key=lambda x: wiki_df[x] if x in wiki_df else 0, reverse=True)
        candidates = candidates[:config.EL_MAX_CAND_NUM]
        if golden_entity not in candidates:
            return
        assert len(candidates) > 0 and len(candidates) <= config.EL_MAX_CAND_NUM
        candidates += [0] * (config.EL_MAX_CAND_NUM - len(candidates))
        candidate_ids = torch.tensor([ent_utils.get_id_from_ent(candidate) for candidate in candidates])

        candidate_to_id = self.candidate_to_id
        pem_candidate_ids = [candidate_to_id[candidate] if candidate in candidate_to_id else 0 for candidate in candidates]
        pem_candidate_ids = torch.tensor(pem_candidate_ids)

        mention_to_id = self.mention_to_id
        mention_id = mention_to_id[mention.mention]
        mention_id = torch.tensor(mention_id)

        # get WLM
        wiki_wlm = self.wiki_wlm
        known_mentions = []
        known_entities = []
        for mention_index in doc.known_mentions:
            mention = doc.mentions[mention_index]
            known_mentions.append(mention)
            known_entities.append(mention.golden_entity)
        wlms = []
        for candidate in candidates:
            score = 0.0
            if (candidate == 0) or (candidate not in wiki_wlm):
                wlms.append(score)
                continue
            for known_entity in known_entities:
                score += wiki_wlm[candidate][known_entity] if known_entity in wiki_wlm[candidate] else 0.0
            wlms.append(score)
        wlms = torch.tensor(wlms)
        if len(known_entities) > 0:
            wlms = wlms / len(known_entities)

        # get context entities
        sorted_known_mentions = sorted(known_mentions, key=lambda x: mention_distance(x, unknown_mention))
        known_entities = [mention.golden_entity for mention in sorted_known_mentions[:config.EL_MAX_KNOWN_ENT_NUM]]
        assert len(known_entities) >= 0 and len(known_entities) <= config.EL_MAX_KNOWN_ENT_NUM
        known_entities_len = torch.tensor(max(1, len(known_entities)))
        known_entities += [0] * (config.EL_MAX_KNOWN_ENT_NUM - len(known_entities))
        known_entities = torch.tensor([ent_utils.get_id_from_ent(ent) for ent in known_entities])

        target = torch.tensor(candidates.index(golden_entity))
        # SIZE: MAX_CAND_NUM, CTXT_LEN * VEC_SIZE, 1
        # ins = (candidate_ids, ctxt_word_vecs, pes, wlms, known_entities, known_entities_len, target)
        ins = (candidate_ids, ctxt_word_vecs, mention_id, pem_candidate_ids, wlms, known_entities, known_entities_len, target)
        ins = tuple(i.to(device) for i in ins)
        self.data.append(ins)

    def train_one_epoch(self, epoch):
        dataloader = DataLoader(self.data, batch_size = config.EL_BATCH_SIZE, shuffle=True)
        model = self.model
        optimizer = self.optimizer
        criterion = self.criterion
        model.train()
        total_loss = 0
        start_time = time.time()
        for candidate_ids, ctxt_word_vecs, mention_id, pem_candidate_ids, wlms, known_entities, known_entities_len, target in dataloader:
            optimizer.zero_grad()
            predict = model((candidate_ids, ctxt_word_vecs, mention_id, pem_candidate_ids, wlms, known_entities, known_entities_len))
            loss = criterion(predict, target)
            total_loss += loss.item()
            loss.backward()
            if self.is_train_feature:
                model.pem.grad[:,:-self.new_entity_num] = 0
                model.wlm_scale.grad[:-self.new_entity_num] = 0
                model.ent_embeddings.weight.grad[:-self.new_entity_num,:] = 0
            optimizer.step()

        cost_time = time.time() - start_time
        logger.info("Epoch {}: total loss = {}, time cost = {}s.".format(epoch, total_loss, cost_time))

    def normalize_f_net(self):
        model = self.model
        f_net = model.f_network
        ps = f_net.parameters()
        for pp in ps:
            if pp.data.norm() > 1.0:
                pp.data = pp.data / pp.data.norm()

    def disambiguation(self, mention, doc, vocab, digest=None):

        word_utils = self.word_utils
        # get context
        unknown_mention = mention
        pos = mention.pos
        mention_to_token = doc.mention_to_token
        pos = [mention_to_token[p] for p in pos if p in mention_to_token]
        assert len(pos) > 0
        logits_list = []

        model = self.model
        model.eval()

        # get candidates
        wiki_df = self.wiki_df
        ent_utils = self.ent_utils
        candidates = [get_origin_entity(candidate) for candidate in mention.candidates]
        candidate_ids = torch.tensor([ent_utils.get_id_from_ent(candidate) for candidate in candidates])

        candidate_to_id = self.candidate_to_id
        pem_candidate_ids = [candidate_to_id[candidate] if candidate in candidate_to_id else 0 for candidate in candidates]
        pem_candidate_ids = torch.tensor(pem_candidate_ids)
        
        mention_to_id = self.mention_to_id
        mention_id = mention_to_id[mention.mention]
        mention_id = torch.tensor(mention_id)
        # get WLM
        wiki_wlm = self.wiki_wlm
        known_mentions = []
        known_entities = []
        for mention_index in doc.known_mentions:
            mention = doc.mentions[mention_index]
            known_mentions.append(mention)
            known_entities.append(mention.golden_entity)
        wlms = []
        for candidate in candidates:
            score = 0.0
            if (candidate == 0) or (candidate not in wiki_wlm):
                wlms.append(score)
                continue
            for known_entity in known_entities:
                score += wiki_wlm[candidate][known_entity] if known_entity in wiki_wlm[candidate] else 0.0
            wlms.append(score)
        wlms = torch.tensor(wlms)
        if len(known_entities) > 0:
            wlms = wlms / len(known_entities)

        # get context entities
        sorted_known_mentions = sorted(known_mentions, key=lambda x: mention_distance(x, unknown_mention))
        known_entities = [mention.golden_entity for mention in sorted_known_mentions[:config.EL_MAX_KNOWN_ENT_NUM]]
        # assert len(known_entities) >= 0 and len(known_entities) <= config.EL_MAX_KNOWN_ENT_NUM
        known_entities_len = torch.tensor(max(1, len(known_entities)))
        known_entities += [0] * (config.EL_MAX_KNOWN_ENT_NUM - len(known_entities))
        known_entities = torch.tensor([ent_utils.get_id_from_ent(ent) for ent in known_entities])

        ins_list = ([], [], [], [], [], [], [])
        for cur_pos in pos:
            cur_pos = random.choice(pos)

            tokens = doc.tokens
            half_ctxt_len = config.EL_CTXT_LEN // 2
            ctxt = []
            left_ctxt = tokens[max(0, cur_pos[0]-half_ctxt_len) :cur_pos[0]]
            ctxt += left_ctxt
            right_ctxt = tokens[cur_pos[1]: min(cur_pos[1]+half_ctxt_len, len(tokens))]
            ctxt += right_ctxt
            ctxt = [vocab[wid] for wid in ctxt]
            ctxt_ids = []
            for word in ctxt:
                if word_utils.contains_word(word):
                    ctxt_ids.append(word_utils.get_id_from_word(word))
            tmp_ctxt = [word_utils.get_word_from_id(wid) for wid in ctxt_ids]
            # print(tmp_ctxt)
            if len(ctxt_ids) == 0:
                ctxt_ids = [1234]
            assert len(ctxt_ids) > 0 and len(ctxt_ids) <= config.EL_CTXT_LEN
            ctxt_ids += [0] * (config.EL_CTXT_LEN - len(ctxt_ids))
            ctxt_word_vecs = word_utils.get_embeddings(torch.tensor(ctxt_ids))

            # ins = (candidate_ids, ctxt_word_vecs, pes, wlms, known_entities, known_entities_len)
            ins = (candidate_ids, ctxt_word_vecs, mention_id, pem_candidate_ids, wlms, known_entities, known_entities_len)
            ins = tuple(i.to(device) for i in ins)
            for i in range(7):
                ins_list[i].append(ins[i])
        
        ins = tuple(torch.stack(feat) for feat in ins_list)
        with torch.no_grad():
            input_features = ins

            # DeepED
            logits = model(input_features).cpu()
            # # Yamada
            # logits = model.mlp(input_features, add_info=(mention.mention ,candidates, self.id_names))

        # DeepED
        max_logits, _ = logits.max(0)
        return max_logits.numpy()

        # # Yamada
        # max_logits, _ = logits.max(0)
        # return max_logits.cpu().numpy()

    def finish(self):
        pass



def main():
    linker = ParamContextModel([], saved_model_path='./model/linking_model.W0.498.ckpt')
    logger.info(linker)

if __name__ == '__main__':
    main()