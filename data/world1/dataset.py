import random, time, json, pickle
from tqdm import tqdm
from collections import defaultdict
from config import config, logger, random_seed
from model import Doc, Mention, Query

dataset_dict_path = './data/world1/dict.pkl'
dataset_entity_path = './data/world1/entity.pkl'
dataset_doc_path = './data/world1/doc.pkl'
dataset_vocab_path = './data/world1/vocab.pkl'
# aida_feature_path = '.\\dataset\\aida_feature_path.pkl'
stopwords_path = '.data/stopwords6.txt'

wiki_dataset_path = './data/world1/wiki_test_dataset.final.tid.pkl'

def load_dataset():
    random.seed(666)
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
    all_candidates = set()

    wiki_test_dataset = {}
    # [new] wiki test dataset
    with open(wiki_dataset_path, 'rb') as fin:
        wiki_data = pickle.load(fin)
    wiki_entity_to_docs = wiki_data['entity_to_docs']
    wiki_docs = wiki_data['docs']

    for ii, raw_entity in enumerate(raw_entities):
        wiki_id = raw_entity[0]
        # 统计所有候选实体
        all_candidates.add(wiki_id)

        mentions = [item[0] for item in raw_entity[5]]
        doc_ids = raw_entity[6]
        docs = {doc_id: doc_info[doc_id] for doc_id in doc_ids}
        # query_entity = QueryEntity(wiki_id, mentions, docs, candidate_dict)
        query_entity = Query()
        query_entity.wiki_id = wiki_id
        query_entity.mentions = mentions

        # [new] wiki test dataset:
        wiki_test_dataset[wiki_id] = []
        for doc_id in wiki_entity_to_docs[wiki_id]:
            raw_doc = wiki_docs[doc_id]
            doc = Doc()
            doc.id = doc_id
            doc.tokens = raw_doc['doc_tokens']
            doc.mention_to_token = {}
            # mention_start, mention_end, wiki_id, mention
            mentions = raw_doc['mentions']
            tmp_mentions = {}
            for mention_start, mention_end, entity, mention_text in mentions:
                doc.mention_to_token[(mention_start, mention_end)] = (mention_start, mention_end)
                if (mention_text, entity) not in tmp_mentions:
                    mention = Mention()
                    mention.mention = mention_text
                    mention.pos.append((mention_start, mention_end))
                    mention.golden_entity = entity
                    if mention.mention in query_entity.mentions:
                        mention.known = False
                        for candidate in candidate_dict[mention.mention]:
                            mention.candidates.append(candidate)
                            all_candidates.add(candidate)
                        if mention.golden_entity not in mention.candidates:
                            print(mention.mention, mention.golden_entity, mention.candidates, query_entity.wiki_id)
                        assert mention.golden_entity in mention.candidates
                        doc.mentions.append(mention)
                        tmp_mentions[(mention_text, entity)] = mention
                    elif mention.golden_entity != query_entity.wiki_id:
                        mention.known = True
                        mention.entity = mention.golden_entity
                        doc.mentions.append(mention)
                        tmp_mentions[(mention_text, entity)] = mention
                else:
                    mention = tmp_mentions[(mention_text, entity)]
                    mention.pos.append((mention_start, mention_end))
            for i, mention in enumerate(doc.mentions):
                if mention.known:
                    doc.known_mentions.append(i)
                else:
                    doc.unknown_mentions.append(i)
                    b+=1
            wiki_test_dataset[wiki_id].append(doc)

        # facc dataset:
        for doc_id, raw_doc in tqdm(docs.items()):
            doc = Doc()
            doc.id = doc_id
            doc.tokens = raw_doc['tokens'].copy()
            doc.token_mentions = raw_doc['token_mentions']
            doc.token_mentions = {k: v for k, v in doc.token_mentions.items() if (k[0] < len(doc.tokens)) and (k[1] <= len(doc.tokens))}
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
                    # 统计所有实体名单
                    # all_entity.add(mention.golden_entity)

                    if mention.mention in query_entity.mentions:
                        mention.known = False
                        for candidate in candidate_dict[mention.mention]:
                            mention.candidates.append(candidate)
                            # 统计所有候选实体
                            all_candidates.add(candidate)

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
            if ii == 1:
                break
    print(a, b)
    random.seed(random_seed)
    # 读入额外的wiki dataset

    print('all candidates #: ', len(all_candidates))
    # return query_entities
    test_dataset = {'facc': test_dataset, 'wiki': wiki_test_dataset}
    return train_dataset, test_dataset, names, all_candidates

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
