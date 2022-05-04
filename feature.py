# coding=utf-8
import tqdm, pickle, math, spacy, re, random, time
import numpy as np
from config import config, logger, device, word2vec, random_seed
from gensim.models import KeyedVectors
from model import EntToVecModel
from utils import WordUtils
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from functools import partial

facc_df_path = './data/dataset/facc_df.pkl'
# tokenizer
sp = spacy.load('en_core_web_sm', disable=['ner', 'parser', 'targger'])
stop_word_path = './data/stopwords6.txt'
stop_word = set()
with open(stop_word_path, 'r', encoding='utf-8') as fin:
    for line in fin:
        stop_word.add(line.strip())

def is_stop_word_or_number(w):
    if len(w) <= 1 or w.lower() in stop_word or w.isnumeric():
        return True
    else:
        return False

torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)


def spacy_tokenize(text):
    text = text.strip()
    if len(text) == 0:
        return []
    text = re.sub(r'\s+', ' ', text)
    tokens = sp(text)
    tokens = [token.text for token in tokens]
    return tokens

def load_facc_df():
    with open(facc_df_path, 'rb') as fin:
        facc_df = pickle.load(fin)
        return facc_df

def wlm_feature(dataset, samples):
    N = 31988897
    facc_df = load_facc_df()
    counts = {}
    for wiki_id, docs in dataset.items():
        # wiki_id = query_entity.wiki_id
        facc_df[wiki_id] = 0
        counts[wiki_id] = {}
        for doc in docs:
            selected = False
            for idx in doc.unknown_mentions:
                mention = doc.mentions[idx]
                if mention.entity == wiki_id:
                    selected = True
            if not selected:
                continue
            # doc中出现了query entity:
            facc_df[wiki_id] += 1
            for idx in doc.known_mentions:
                mention = doc.mentions[idx]
                # other_entity = mention.golden_entity
                other_entity = mention.entity
                if other_entity in dataset or other_entity not in facc_df:
                    continue
                if other_entity not in counts[wiki_id]:
                    counts[wiki_id][other_entity] = 0
                counts[wiki_id][other_entity] += 1
    feature = {}
    for entity, v in counts.items():
        feature[entity] = {}
        for other_entity, count in v.items():
            # print(facc_df[entity], facc_df[other_entity], count)
            v = 1 - (math.log(max(facc_df[entity] * samples[entity], facc_df[other_entity])) - math.log(count  * samples[entity])) / (math.log(N) - math.log(min(facc_df[entity] * samples[entity], facc_df[other_entity])))
            feature[entity][other_entity] = v
    return feature

def embedding_feature(dataset, vocab, names, init_embeddings=None, max_epoch=None, lr=None):
    train_data = []
    ent_to_id = {}
    neg_word_utils = WordUtils(task='EMB')
    pos_word_utils = WordUtils()
    # 收集实体出现的上下文
    logger.info('-- collecting context')
    for wiki_id, docs in dataset.items():
        # wiki_id = query_entity.wiki_id
        count = 0
        if wiki_id not in ent_to_id:
            ent_to_id[wiki_id] = len(ent_to_id)
        for doc in docs:
            tokens = doc.tokens
            mention_to_token = {v: k for k,v in doc.token_mentions.items()}
            for idx in doc.unknown_mentions:
                mention = doc.mentions[idx]
                if mention.entity != wiki_id:
                    continue
                for pos in mention.pos:
                    if pos not in mention_to_token:
                        continue
                    token_begin, token_end = mention_to_token[pos]
                    # print(tokens[token_begin: token_end])
                    half_ctxt_len = config.LOCAL_E2V_CTXT_LEN
                    left_ctxt = tokens[max(0, token_begin - half_ctxt_len): token_begin]
                    right_ctxt = tokens[token_end: min(token_end + half_ctxt_len, len(tokens))]
                    ctxt = [] + left_ctxt + right_ctxt
                    ctxt = [vocab[wid] for wid in ctxt]
                    ctxt = [word for word in ctxt if (not is_stop_word_or_number(word)) and pos_word_utils.contains_word(word)] 
                    if len(ctxt) > 0:
                        train_data.append((ent_to_id[wiki_id], ctxt))
                    count += 1
        # print(count)
    if len(train_data) > 60000:
        random.shuffle(train_data)
        train_data = train_data[:60000]
    logger.info('traning data size: {}'.format(len(train_data)))
    

    # 如果没有设定，根据实体名字设定初始embedding
    logger.info('-- init embedding')
    if init_embeddings is None:
        # logger.info("Build init embedding from entity names.")
        init_embeddings = {}
        # word2vec_path = './data/GoogleNews-vectors-negative300.bin'
        # w2v = KeyedVectors.load_word2vec_format(word2vec_path, binary=True)
        w2v = word2vec.get()
        # logger.info("Raw word2vec data loaded.")
        for wiki_id in dataset:
            # wiki_id = query_entity.wiki_id
            mentions = names[wiki_id]
            init_vec = np.zeros(config.EMBEDDING_SIZE)
            num_words_title = 0
            for name in mentions:
                name_tokens = spacy_tokenize(name)
                for token in name_tokens:
                    if token not in w2v.vocab:
                        continue
                    num_words_title += 1
                    init_vec += w2v[token]
            if num_words_title > 0:
                init_vec /= num_words_title
            else:
                print('random init entity embedding.')
                init_vec = np.random.randn(config.EMBEDDING_SIZE)
            init_embeddings[wiki_id] = init_vec

    # 初始化embedding矩阵和各种模型信息
    if lr is None:
        lr = config.LOCAL_E2V_LR
    ent_init_embeddings = np.random.randn(len(ent_to_id), config.EMBEDDING_SIZE)
    for k, v in ent_to_id.items():
        # ent_init_embeddings[v] = init_embeddings[k]
        ent_init_embeddings[v] = np.random.randn(config.EMBEDDING_SIZE)
    model = EntToVecModel(len(ent_to_id), torch.tensor(ent_init_embeddings).float()).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = nn.MultiMarginLoss(margin=0.1)
    model.train()

    print('init embedding now')
    embedding_data = model.ent_embeddings.weight.data
    yield 0, {k:embedding_data[v] for k, v in ent_to_id.items()}


    # 我们需要使用neg_word_utils随机产生neg word
    total_inst_num = 0
    cfn = partial(e2v_collate_fn,neg_word_utils=neg_word_utils, pos_word_utils=pos_word_utils)
    dataloader = DataLoader(train_data, batch_size = 100, shuffle=True, collate_fn=cfn)
    logger.info('-- begin training')
    if max_epoch is None:
        max_epoch = config.LOCAL_E2V_EPOCH
    for epoch in range(max_epoch):
        start_time = time.time()
        total_loss = 0.0
        for batch in dataloader:
            # print(batch)
            ctxt_word_vecs, ent_idxes, targets = batch
            predicts = model((ctxt_word_vecs, ent_idxes))
            # print(predicts.size(), targets.size())
            loss = criterion(predicts, targets)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

        cost_time = time.time() - start_time
        logger.info("-- Epoch {}: total loss = {}, time cost = {}s.".format(epoch, total_loss, cost_time))
        # exit(0)
        
        embedding_data = model.ent_embeddings.weight.data
        yield epoch+1, {k:embedding_data[v] for k, v in ent_to_id.items()}
        


def e2v_collate_fn(batch, neg_word_utils, pos_word_utils):
    ctxt_word_vecs = []
    ent_idxes = []
    targets = []
    config.BATCH_SIZE = len(batch)
    for local_ent_idx, positive_words in batch:
        # positive_words = []
        # for word in line_words:
        #     if (not is_stop_word_or_number(word)) and pos_word_utils.contains_word(word):
        #         positive_words.append(word)
        # print(positive_words)
        if len(positive_words) == 0:
            positive_words.append(neg_word_utils.get_word_from_id(neg_word_utils.random_w_id()[0]))
        local_targets = []
        # TODO：需要直接转为vec
        local_ctxt_word_ids = neg_word_utils.random_w_id(k = config.NUM_WORDS_PER_ENT* config.NUM_NEG_WORDS)
        local_ctxt_word_vecs = neg_word_utils.get_embeddings(torch.tensor(local_ctxt_word_ids))

        for i in range(config.NUM_WORDS_PER_ENT):
            select_pos_w = random.choice(positive_words)
            # TODO:需要直接转为vec
            select_pos_w_id = pos_word_utils.get_id_from_word(select_pos_w)
            select_pos_w_vec = pos_word_utils.get_embeddings(torch.tensor(select_pos_w_id))
            grd_trth = random.randint(0, config.NUM_NEG_WORDS-1)

            local_ctxt_word_vecs[i*config.NUM_NEG_WORDS+grd_trth] = select_pos_w_vec
            local_targets.append(grd_trth)

        ctxt_word_vecs.append(local_ctxt_word_vecs)
        ent_idxes.append(local_ent_idx)
        targets.append(local_targets)

    ctxt_word_vecs = torch.stack(ctxt_word_vecs).view(-1, config.EMBEDDING_SIZE).to(device)
    ent_idxes = torch.tensor(ent_idxes).to(device)
    targets = torch.tensor(targets).view(-1).to(device)

    # print(ctxt_word_vecs.size(), ent_idxes.size(), targets.size())

    return ctxt_word_vecs, ent_idxes, targets


def pem_feature(dataset):
    counts= {}
    for wiki_id, docs in dataset.items():
        # wiki_id = query_entity.wiki_id
        for doc in docs:
            for idx in doc.unknown_mentions:
                mention = doc.mentions[idx]
                name = mention.mention
                entity = mention.entity
                if name not in counts:
                    counts[name] = {}
                if entity not in counts[name]:
                    counts[name][entity] = 0
                counts[name][entity] += 1
    return counts

def build_feature(dataset):
    pass

def main():
    # unit test
    pass

if __name__ == '__main__':
    main()