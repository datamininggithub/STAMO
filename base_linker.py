'''
    some EL methods using basic features
'''
import pickle, random, math
from collections import defaultdict
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from model import EntToVecModel
from utils import *
from feature import wlm_feature, pem_feature, embedding_feature


TOTAL_YAGO_ENTITIES = 2651987
aida_score_cache_path = '.\\dataset\\aida_score_cache.pkl'
struct_model_path = '.\\dataset\\struct_model.pt'
wiki_df_path = './data/wiki/wiki_df.txt'

wiki_dict_path = './data/wiki/wiki_dict.txt'

torch.manual_seed(666)
torch.cuda.manual_seed(666)
np.random.seed(666)
random.seed(666)

def load_wiki_df():
    wiki_df = {}
    with open(wiki_df_path, 'r', encoding='utf-8') as fin:
        for line in tqdm(fin):
            row = line.strip().split('\t')
            wiki_df[row[0]] = int(row[1])
    return wiki_df

def load_wiki_wlm():
    if config.TASK == 'IER':
        wiki_wlm_path = './data/wiki/wiki_wlm.IER.pkl'
    elif config.TASK == 'IE':
        wiki_wlm_path = './data/wiki/wiki_wlm.IE.pkl'
    elif config.TASK == 'W1':
        wiki_wlm_path = './data/wiki/wiki_wlm.W1.pkl'
    else:
        wiki_wlm_path = './data/wiki/wiki_wlm.pkl'
    print('wlm data path: {}'.format(wiki_wlm_path))
    with open(wiki_wlm_path, 'rb') as fin:
        wiki_wlm = pickle.load(fin)
        print(len(wiki_wlm))
        return wiki_wlm

def load_wiki_dict():
    wiki_dict = {}
    with open(wiki_dict_path, 'r', encoding='utf-8') as fin:
        for line in tqdm(fin):
            row = line.strip().split('\t')
            if len(row) % 2 == 0:
                print(line)
                continue
            mention = row[0]
            wiki_dict[mention] = {}
            for i in range(len(row)//2):
                wiki_dict[mention][row[i*2+1]] = int(row[i*2+2])
    return wiki_dict

def get_origin_entity(entity):
    if entity.startswith('new:'):
        return entity[4:]
    else:
        return entity

class Linker:
    def __init__(self, query_sets):
        pass

    def disambiguation(self, mention, doc, wid_to_word, digest=None):
        candidates = mention.candidates
        return [0.0] * len(candidates)

    def finish(self):
        pass

class RandomLinker(Linker):
    def __init__(self, query_sets):
        super().__init__(query_sets)
        random.seed(666)

    def disambiguation(self, mention, doc, wid_to_word, digest=None):
        candidates = mention.candidates
        return [random.random() for _ in candidates]

class WLMLinker(Linker):
    def __init__(self, query_sets):
        super().__init__(query_sets)
        self.wiki_wlm = load_wiki_wlm()
        # for query_entity in query_sets:
        #     self.wiki_wlm[query_entity.wiki_id] = self.wiki_wlm[query_entity.init_wiki_id]

    def disambiguation(self, mention, doc, wid_to_word, digest=None):
        logits = []
        for candidate in mention.candidates:
            score = 0
            wlms = self.wiki_wlm[candidate]
            for mention_index in doc.known_mentions:
                mention = doc.mentions[mention_index]
                other_entity = mention.golden_entity
                score += wlms[other_entity] if other_entity in wlms else 0.0
            logits.append(score)
        return logits

    def update_feature(self, dataset, vocab, names, samples, valid_func=None):
        f = wlm_feature(dataset, samples)
        for k, v in f.items():
            self.wiki_wlm[k] = v

class PEMPopularityLinker(Linker):
    def __init__(self, query_sets):
        super().__init__(query_sets)
        self.wiki_dict = load_wiki_dict()

    def disambiguation(self, mention, doc, wid_to_word, digest=None):
        logits = []
        counts = self.wiki_dict[mention.mention] if mention.mention in self.wiki_dict else {}
        for candidate in mention.candidates:
            candidate = get_origin_entity(candidate)
            count = counts[candidate] if candidate in counts else 0
            logits.append(count)
        return logits

    def update_feature(self, dataset, vocab, names, samples, valid_func=None):
        f = pem_feature(dataset)
        wiki_dict = self.wiki_dict
        for wiki_id in dataset:
            mentions = names[wiki_id]
            for mention in mentions:
                # if mention not in wiki_dict or wiki_id not in wiki_dict[mention]:
                if mention not in wiki_dict:
                    continue
                wiki_dict[mention][wiki_id] = 0.0
                if mention not in f or wiki_id not in f[mention]:
                    continue

                '''
                x = f[mention][wiki_id]
                y = 0.0
                z = 0.0
                for entity in wiki_dict[mention]:
                    if entity == wiki_id or entity not in f[mention]:
                        continue
                    y += wiki_dict[mention][entity]
                    z += f[mention][entity]
                score = y / z * x if z > 0 else 0.0
                wiki_dict[mention][wiki_id] = score
                '''
                a = 0.0
                for entity in wiki_dict[mention]:
                    a += wiki_dict[mention][entity]
                x = 0.0
                y = 0.0
                for entity in f[mention]:
                    if entity == wiki_id:
                        x += f[mention][entity]
                    y += f[mention][entity]
                p = x / y if y > 0 else 0.0
                if p > 0  and p < 1:
                    score = p / (1-p) * a
                    wiki_dict[mention][wiki_id] = score

class EntityRelatenessLinker(Linker):
    def __init__(self, query_sets):
        super().__init__(query_sets)
        ent_utils_path = config.ent_utils_path()
        with open(ent_utils_path, 'rb') as fin:
            data = pickle.load(fin)
            self.ent_utils = data['ent_utils']
            self.ent_init_embeddings = data['ent_init_embeddings']
        model_path = config.ent_model_path()
        model = EntToVecModel(self.ent_utils.get_total_ent_num())
        model.load_state_dict(torch.load(model_path))
        # model = model.to(device)
        model.eval()
        self.model = model

    def disambiguation(self, mention, doc, wid_to_word, digest=None):
        with torch.no_grad():
            logits = []
            other_entities = []
            for mention_index in doc.known_mentions:
                known_mention = doc.mentions[mention_index]
                other_entity = known_mention.golden_entity
                idx = self.ent_utils.get_id_from_ent(other_entity)
                other_entities.append(idx)
            other_entities = torch.tensor(other_entities)
            if len(other_entities) == 0:
                return [random.random() for _ in mention.candidates]
            other_vecs = self.model.embedding(other_entities).numpy()
            
            for candidate in mention.candidates:
                candidate = get_origin_entity(candidate)
                e1_index = torch.tensor([self.ent_utils.get_id_from_ent(candidate)])
                e1_vec = self.model.embedding(e1_index).cpu().numpy()[0]
                sims = np.dot(other_vecs, e1_vec.T)
                score = np.sum(sims)
                logits.append(score)
            return logits

    def update_feature(self, dataset, vocab, names, samples, valid_func=None):
        init_embeddings = None
        init_embeddings = {}
        for wiki_id in dataset:
            init_embeddings[wiki_id] = self.ent_init_embeddings[self.ent_utils.get_id_from_ent(wiki_id)]
        #
        wid_to_word = {v:k for k, v in vocab.items()}
        # tmp_dataset = [query_entity]
        model = self.model
        embedding_data = model.ent_embeddings.weight.data
        ent_utils = self.ent_utils
        for epoch, f in embedding_feature(dataset, wid_to_word, names, init_embeddings=init_embeddings, max_epoch=800):
            for k, v in f.items():
                embedding_data[ent_utils.get_id_from_ent(k)] = v
            if epoch >= 300:
                if epoch % 10 == 0:
                    valid_func(linker=self)



class PopularityLinker(Linker):
    def __init__(self, query_sets):
        super().__init__(query_sets)
        self.wiki_df = load_wiki_df()
        for query_entity in query_sets:
            self.wiki_df[query_entity.wiki_id] = self.wiki_df[query_entity.init_wiki_id] if query_entity.init_wiki_id in self.wiki_df else 0

    def disambiguation(self, mention, doc, wid_to_word, digest=None):
        logits = []
        for candidate in mention.candidates:
            pop = self.wiki_df[candidate] if candidate in self.wiki_df else 0
            logits.append(pop)
        return logits

class IdealAIDALinker(Linker):
    def __init__(self, query_sets, cache_path = None):
        super().__init__(query_sets)
        self.load_feature()
        for query_entity in query_sets:
            self.entity_keyphrases[query_entity.wiki_id] = self.entity_keyphrases[query_entity.init_wiki_id]
        if cache_path is not None:
            self.cache_path = cache_path
        else:
            self.cache_path = aida_score_cache_path
        if self.cached:
            print('feature cached')
            with open(self.cache_path, 'rb') as fin:
                self.cache_score = pickle.load(fin)
        else:
            self.cache_score = {}

    def finish(self):
        if not self.cached:
            with open(self.cache_path, 'wb') as fout:
                pickle.dump(self.cache_score, fout)


    def load_feature(self):
        with open(sample_aida_feature_path, 'rb') as fin:
            aida_info = pickle.load(fin)
        self.entity_keyphrases = aida_info['entity_keyphrases']
        self.keyphrase_tokens = aida_info['keyphrase_tokens']
        self.token_to_count = aida_info['token_to_count']
        self.dict = aida_info['aida_dict']
        self.stopwords = set()
        with open(aida_stopwords_path, 'r', encoding='utf-8') as fin:
            for line in fin:
                word = line.strip()
                self.stopwords.add(word)

    def disambiguation(self, mention, candidates, doc, vocab):
        mention_index = set()
        for token_mention in doc.token_mentions:
            for i in range(token_mention[0], token_mention[1]):
                mention_index.add(i)
        inversed_doc = defaultdict(list)
        for i, token in enumerate(doc.tokens):
            if i not in mention_index:
                inversed_doc[token].append(i)
        scores = []
        for candidate in candidates:
            score = self.score(mention, candidate, inversed_doc, vocab, doc.id)
            scores.append(score)
        return scores

    def score(self, mention, entity, inversed_doc, vocab, doc_id):
        if entity not in self.entity_keyphrases:
            # print('Info: entity {} not found in aida features.'.format(entity))
            return 0.0
        score = 0.0
        keyphrases = self.entity_keyphrases[entity]
        for keyphrase in keyphrases:
            if self.cached:
                if (keyphrase, doc_id) not in self.cache_score:
                    sim = 0.0
                else:
                    sim = self.cache_score[(keyphrase, doc_id)]
            else:
                sim = self.doc_keyphrase_sim(inversed_doc, keyphrase, vocab)
                if sim > 0:
                    self.cache_score[(keyphrase, doc_id)] = sim
            score += sim
        return score

    def get_keyword_score(self, keyword):
        if keyword in self.token_to_count:
            count = self.token_to_count[keyword]
            idf = TOTAL_YAGO_ENTITIES / count
            return max(0.0, math.log(idf) / math.log(2))
        else:
            return 0.0


    def doc_keyphrase_sim(self, inversed_doc, keyphrase, vocab):
        sim = 0.0
        positions = {}
        all_keyword_total_score = 0.0
        common_keyword_total_score = 0.0
        tokens = self.keyphrase_tokens[keyphrase]
        for token_id in tokens:
            token_id = token_id[0]
            keyword_score = self.get_keyword_score(token_id)
            all_keyword_total_score += keyword_score
            if token_id not in self.dict:
                # print('Info: token_id {} not fount in aida dict.'.format(token_id))
                continue
            keyword = self.dict[token_id].lower()
            if keyword in self.stopwords:
                continue
            if keyword not in vocab.vocab:
                continue
            keyword_id = vocab.vocab[keyword]
            if keyword_id in inversed_doc:
                common_keyword_total_score += keyword_score
                positions[keyword_id] = inversed_doc[keyword_id]
        min_cover = self.calculate_min_cover(positions)
        if common_keyword_total_score > 0:
            sim = all_keyword_total_score * (len(positions) / min_cover) * ((common_keyword_total_score / all_keyword_total_score)**2)
            # sim = common_keyword_total_score
        return max(0.0, sim)

    def calculate_min_cover(self, positions):
        positions = [v for k, v in positions.items()]
        if len(positions) == 0:
            return -1
        if len(positions) == 1:
            return 1
        priority_buffer = []
        for i, ps in enumerate(positions):
            for p in ps:
                priority_buffer.append((i, p))
        priority_buffer = sorted(priority_buffer, key=lambda x:x[1])
        
        symbol_count = len(positions)
        covered_symbol_count = 0
        cur = 0
        covered_occur = [-1] * symbol_count
        while (covered_symbol_count < symbol_count):
            symbol = priority_buffer[cur]
            cur += 1
            if covered_occur[symbol[0]] == -1:
                covered_symbol_count += 1
            covered_occur[symbol[0]] = symbol[1]

        min_selected_occur = covered_occur[0]
        max_selected_occur = covered_occur[1]
        for occur in covered_occur:
            min_selected_occur = min(occur, min_selected_occur)
            max_selected_occur = max(occur, max_selected_occur)
        min_cover_length = max_selected_occur - min_selected_occur + 1
        while min_cover_length > symbol_count and cur < len(priority_buffer):
            symbol = priority_buffer[cur]
            cur += 1
            max_selected_occur = symbol[1]
            if (covered_occur[symbol[0]] == min_selected_occur):
                covered_occur[symbol[0]] = symbol[1]
                min_selected_occur = covered_occur[0]
                for occur in covered_occur:
                    min_selected_occur = min(occur, min_selected_occur)
            else:
                covered_occur[symbol[0]] = symbol[1]
            cover_length = max_selected_occur - min_selected_occur + 1
            if cover_length < min_cover_length:
                min_cover_length = cover_length
        return min_cover_length


class StructLinker(Linker):
    def __init__(self, query_sets, cached=False, cache_path = None):
        super().__init__(query_sets, cached)
        self.load_feature()
        for query_entity in query_sets:
            self.entity_keyphrases[query_entity.wiki_id] = self.entity_keyphrases[query_entity.init_wiki_id]
        self.wiki_wlm = load_wiki_wlm()
        for query_entity in query_sets:
            self.wiki_wlm[query_entity.wiki_id] = self.wiki_wlm[query_entity.init_wiki_id]
        self.wiki_df = load_wiki_df()
        for query_entity in query_sets:
            self.wiki_df[query_entity.wiki_id] = self.wiki_df[query_entity.init_wiki_id]

        if cache_path is not None:
            self.cache_path = cache_path
        else:
            self.cache_path = aida_score_cache_path
        if self.cached:
            print('feature cached')
            with open(self.cache_path, 'rb') as fin:
                self.cache_score = pickle.load(fin)
            self.model = torch.load(struct_model_path)
        else:
            self.cache_score = {}
            self.model = SimpleClassifier()

        self.dataset = StructDataset()
        

    def load_feature(self):
        with open(aida_feature_path, 'rb') as fin:
            aida_info = pickle.load(fin)
        self.entity_keyphrases = aida_info['entity_keyphrases']
        self.keyphrase_tokens = aida_info['keyphrase_tokens']
        self.token_to_count = aida_info['token_to_count']
        self.dict = aida_info['aida_dict']
        self.stopwords = set()
        with open(aida_stopwords_path, 'r', encoding='utf-8') as fin:
            for line in fin:
                word = line.strip()
                self.stopwords.add(word)

    def add_train_instance(self, mention, candidates, golden_entity, doc, vocab):
        x = []
        features = []
        for candidate in candidates:
            pop = self.wiki_df[candidate] if candidate in self.wiki_df else 0
            features.append(pop)
        features = F.normalize(torch.tensor(features, dtype=torch.float), p=1, dim=0)
        x.append(features)
        features = []
        for candidate in candidates:
            score = 0
            wlms = self.wiki_wlm[candidate]
            for mention_index in doc.known_mentions:
                known_mention = doc.mentions[mention_index]
                other_entity = known_mention.golden_entity
                score += wlms[other_entity] if other_entity in wlms else 0.0
            features.append(score)
        features = torch.tensor(features, dtype=torch.float)
        # features = F.normalize(features, p=1, dim=0)
        if len(doc.known_mentions) > 0:
            features = features / math.sqrt(len(doc.known_mentions))
        x.append(features)
        features = []
        for candidate in candidates:
            score = self.score(mention, candidate, doc.id)
            features.append(score)
        features = torch.tensor(features, dtype=torch.float)
        features = F.normalize(features, p=1, dim=0)
        x.append(features)
        x = torch.transpose(torch.stack(x), 0, 1)
        for i, candidate in enumerate(candidates):
            if candidate == golden_entity:
                y = torch.tensor(i)
        self.dataset.add((x, y))

    def disambiguation(self, mention, candidates, doc, vocab):
        self.model.eval()
        x = []
        features = []
        for candidate in candidates:
            pop = self.wiki_df[candidate] if candidate in self.wiki_df else 0
            features.append(pop)
        features = F.normalize(torch.tensor(features, dtype=torch.float), p=1, dim=0)
        x.append(features)
        features = []
        for candidate in candidates:
            score = 0
            wlms = self.wiki_wlm[candidate]
            for mention_index in doc.known_mentions:
                known_mention = doc.mentions[mention_index]
                other_entity = known_mention.golden_entity
                score += wlms[other_entity] if other_entity in wlms else 0.0
            features.append(score)
        features = torch.tensor(features, dtype=torch.float)
        # features = F.normalize(features, p=1, dim=0)
        if len(doc.known_mentions) > 0:
            features = features / math.sqrt(len(doc.known_mentions))
        x.append(features)
        features = []
        for candidate in candidates:
            score = self.score(mention, candidate, doc.id)
            features.append(score)
        features = torch.tensor(features, dtype=torch.float)
        features = F.normalize(features, p=1, dim=0)
        # features = features / math.sqrt(len(doc.known_mentions))
        x.append(features)
        x = torch.transpose(torch.stack(x), 0, 1)
        with torch.no_grad():
            y = self.model(x)
            score = y.numpy()
            return score

    def score(self, mention, entity, doc_id):
        if entity not in self.entity_keyphrases:
            return 0.0
        score = 0.0
        keyphrases = self.entity_keyphrases[entity]
        for keyphrase in keyphrases:
            if (keyphrase, doc_id) not in self.cache_score:
                sim = 0.0
            else:
                sim = self.cache_score[(keyphrase, doc_id)]
            score += sim
        return score

    def train(self):
        criterion = nn.CrossEntropyLoss()
        train_dataloader = DataLoader(self.dataset, batch_size = 8, collate_fn=struct_collate_fn)
        model = self.model
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-3)
        model.train()
        max_epoch = 30
        for _ in range(max_epoch):
            total_loss = 0
            for i, (batch_x, batch_y) in enumerate(train_dataloader):
                optimizer.zero_grad()
                logits = model(batch_x).squeeze()
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                if i % 100 == 0:
                    print(i, model.linear.weight.data)
            print(total_loss)

    def finish(self):
        if not self.cached:
            with open(self.cache_path, 'wb') as fout:
                pickle.dump(self.cache_score, fout)
            torch.save(self.model, struct_model_path)

class StructDataset(Dataset):
    def __init__(self):
        self.data = []
    def __getitem__(self, index):
        return self.data[index]
    def __len__(self):
        return len(self.data)
    def add(self, instance):
        self.data.append(instance)

def struct_collate_fn(batch):
    max_len = 0
    for x, y in batch:
        max_len = max(x.size(0), max_len)
    collated_x = []
    collated_y = []
    for x, y in batch:
        tmp = F.pad(x, pad=(0, 0, 0, max_len-x.size(0)), mode='constant', value=0)
        collated_x.append(tmp)
        collated_y.append(y)
    collated_x = torch.stack(collated_x)
    collated_y = torch.stack(collated_y)
    return (collated_x, collated_y)



class SimpleClassifier(nn.Module):
    def __init__(self):
        super(SimpleClassifier, self).__init__()
        self.linear = nn.Linear(3, 1)
        self.linear.weight.data.uniform_(0.1, 1)

    def forward(self, input):
        output = self.linear(input)
        return output

if __name__=='__main__':
    linker = EntityRelatenessLinker([])