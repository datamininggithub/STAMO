import random, time, json, pickle
from tqdm import tqdm
from collections import defaultdict
from config import config, logger
from model import Doc, Mention, Query

dataset_dict_path = './data/world0/dict.pkl'
dataset_entity_path = './data/world0/entity.pkl'
dataset_doc_path = './data/world0/doc.pkl'
dataset_vocab_path = './data/world0/vocab.pkl'
# aida_feature_path = '.\\dataset\\aida_feature_path.pkl'
stopwords_path = '.data/stopwords6.txt'

world1_path = './data/world1/entity.pkl'
with open(world1_path, 'rb') as fin:
    world1 = pickle.load(fin)
    world1 = set([item[0] for item in world1])

'''
class QueryEntity:
    def __init__(self, wiki_id, mentions, raw_docs, candidate_dict):
        assert wiki_id not in world1
        self.init_wiki_id = wiki_id
        self.wiki_id = "new:" + wiki_id
        self.mentions = mentions
        self.docs = []
        self.train_docs = []
        self.test_docs = []
        self.a = 0
        self.b = 0
        for doc_id, raw_doc in tqdm(raw_docs.items()):
            doc = Doc()
            doc.id = doc_id
            # doc.content = raw_doc['content']
            # doc.init_content()
            doc.tokens = raw_doc['tokens']
            doc.token_mentions = raw_doc['token_mentions']
            doc.mention_to_token = {}
            for k, v in doc.token_mentions.items():
                doc.mention_to_token[v] = k
            tmp_mentions = {}
            for info in raw_doc['info']:
                if (info[0], info[3]) not in tmp_mentions:
                    mention = Mention()
                    mention.mention = info[0]
                    mention.pos.append((info[1], info[2]))
                    mention.golden_entity = self.wiki_id if info[3] == self.init_wiki_id else info[3]
                    if mention.golden_entity in world1:
                        # print('#', mention.mention, mention.golden_entity, self.init_wiki_id)
                        continue
                    if mention.mention in self.mentions:
                        mention.known = False
                        for candidate in candidate_dict[mention.mention]:
                            if candidate in world1:
                                continue
                            if candidate == self.init_wiki_id:
                                candidate = self.wiki_id
                            mention.candidates.append(candidate)
                        if mention.golden_entity not in mention.candidates:
                            print('*', mention.mention, mention.golden_entity, self.init_wiki_id, mention.candidates)
                        assert mention.golden_entity in mention.candidates
                        doc.mentions.append(mention)
                        tmp_mentions[(info[0], info[3])] = mention
                    elif mention.golden_entity != self.wiki_id:
                        mention.known = True
                        mention.entity = mention.golden_entity
                        doc.mentions.append(mention)
                        tmp_mentions[(info[0], info[3])] = mention
                else:
                    mention = tmp_mentions[(info[0], info[3])]
                    mention.pos.append((info[1], info[2]))
            for i, mention in enumerate(doc.mentions):
                if mention.known:
                    doc.known_mentions.append(i)
                else:
                    hit = False
                    for pos in mention.pos:
                        if pos in doc.mention_to_token:
                            hit = True
                    if hit:
                        self.a+=1
                        doc.unknown_mentions.append(i)
                    self.b+=1
            if len(doc.unknown_mentions) > 0:
                self.docs.append(doc)
        shuffled_doc_ids = list(range(len(self.docs)))
        random.shuffle(shuffled_doc_ids)
        self.test_docs = shuffled_doc_ids[:200]
        self.train_docs = shuffled_doc_ids[200:]

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
'''

def load_dataset():
    a, b = 0 ,0 
    candidate_dict = load_dict()
    with open(dataset_entity_path, 'rb') as fin:
        raw_entities = pickle.load(fin)
        print('world 1: load entity info.')
    with open(dataset_doc_path, 'rb') as fin:
        doc_info = pickle.load(fin)
        print('world 1: load doc info.')
    query_entities = []
    train_dataset = {}
    test_dataset = {}
    names = {}

    for ii, raw_entity in enumerate(raw_entities):
        wiki_id = raw_entity[0]
        mentions = [item[0] for item in raw_entity[5]]
        doc_ids = raw_entity[6]
        docs = {doc_id: doc_info[doc_id] for doc_id in doc_ids}
        # query_entity = QueryEntity(wiki_id, mentions, docs, candidate_dict)
        query_entity = Query()
        query_entity.wiki_id = wiki_id
        query_entity.mentions = mentions
        for doc_id, raw_doc in tqdm(docs.items()):
            doc = Doc()
            doc.id = doc_id
            doc.tokens = raw_doc['tokens']
            doc.token_mentions = raw_doc['token_mentions']
            doc.mention_to_token = {}
            for k, v in doc.token_mentions.items():
                doc.mention_to_token[v] = k
            tmp_mentions = {}
            # mention, byte_begin, byte_end, entity
            for info in raw_doc['info']:
                if (info[0], info[3]) not in tmp_mentions:
                    mention = Mention()
                    mention.mention = info[0]
                    mention.pos.append((info[1], info[2]))
                    mention.golden_entity = info[3]
                    if mention.golden_entity in world1:
                        # print('#', mention.mention, mention.golden_entity, self.init_wiki_id)
                        continue
                    if mention.mention in query_entity.mentions:
                        mention.known = False
                        for candidate in candidate_dict[mention.mention]:
                            if candidate in world1:
                                continue
                            mention.candidates.append(candidate)
                        if mention.golden_entity not in mention.candidates:
                            print(mention.mention, mention.golden_entity, mention.candidates)
                        assert mention.golden_entity in mention.candidates
                        doc.mentions.append(mention)
                        tmp_mentions[(info[0], info[3])] = mention
                    # 已知实体中不应该包含query entity
                    elif mention.golden_entity != query_entity.wiki_id:
                        mention.known = True
                        mention.entity = mention.golden_entity
                        doc.mentions.append(mention)
                        tmp_mentions[(info[0], info[3])] = mention
                else:
                    mention = tmp_mentions[(info[0], info[3])]
                    mention.pos.append((info[1], info[2]))
            for i, mention in enumerate(doc.mentions):
                if mention.known:
                    doc.known_mentions.append(i)
                else:
                    hit = False
                    for pos in mention.pos:
                        if pos in doc.mention_to_token:
                            hit = True
                    if hit:
                        a+=1
                        doc.unknown_mentions.append(i)
                    b+=1
            if len(doc.unknown_mentions) > 0:
                query_entity.docs.append(doc)
        shuffled_doc_ids = list(range(len(query_entity.docs)))
        random.shuffle(shuffled_doc_ids)
        # query_entity.test_docs = shuffled_doc_ids[:200]
        # query_entity.train_docs = shuffled_doc_ids[200:]
        train_dataset[wiki_id] = []
        test_dataset[wiki_id] = []
        names[wiki_id] = query_entity.mentions
        for idx in shuffled_doc_ids[:200]:
            test_dataset[wiki_id].append(query_entity.docs[idx])
        for idx in shuffled_doc_ids[200:]:
            train_dataset[wiki_id].append(query_entity.docs[idx])
        print('#' + wiki_id)
        if config.UNIT_TEST:
            logger.info('Unit test: we only read one query entity.')
            if ii == 3:
                break
    print(a, b)
    # return query_entities
    return train_dataset, test_dataset, names

def load_dict():
    with open(dataset_dict_path, 'rb') as fin:
        candidate_dict = pickle.load(fin)
    return candidate_dict

def load_vocab():
    with open(dataset_vocab_path, 'rb') as fin:
        vocab = pickle.load(fin)
    return vocab

def main():
    # build_dataset()
    # sample_dataset()
    # sample_aida_feature()
    dataset = load_dataset()
    print(len(dataset))
    vocab = load_vocab()
    wid_to_word = {v:k for k, v in vocab.items()}
    for doc in dataset[0].docs:
        tokens = doc.tokens
        print(tokens)
        tokens = [wid_to_word[token] for token in tokens]
        print(tokens)


    
    

if __name__ == '__main__':
    main()
