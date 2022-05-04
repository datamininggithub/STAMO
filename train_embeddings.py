# coding=utf-8
import pickle, hashlib, os, spacy, re, random, time, sys, logging
import numpy as np
from tqdm import tqdm
from utils import WordUtils, EntityUtils, HyberReader
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import EntToVecModel
from config import config, device

task = 'IE'
relateness_data_path = './data/dataset/relateness.{}.pkl'.format(task)
wiki_ids_path = './data/wiki/ids.txt'
log_path = './log.{}.txt'.format(task)
torch.manual_seed(666)
torch.cuda.manual_seed(666)
np.random.seed(666)
random.seed(666)
config.TASK = "EMB"

# logger

logger = logging.getLogger(__name__)
logger.setLevel(level = logging.INFO)
file_handler = logging.FileHandler(log_path)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(message)s"))
logger.addHandler(file_handler)

std_handler = logging.StreamHandler(sys.stderr)
std_handler.setLevel(logging.INFO)
std_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(message)s"))
logger.addHandler(std_handler)

# from config import logger

################################################################################################
#                                      Global Data

# data = {'valid_queries': valid_queries, 'test_queries': test_queries, 'entities': entities}
with open(relateness_data_path, 'rb') as fin:
    relateness_data = pickle.load(fin)
relateness_entities = relateness_data['entities']
relateness_valid_queries = relateness_data['valid_queries']
relateness_test_queries = relateness_data['test_queries']

sorted_entities = sorted(list(relateness_entities))
digest = hashlib.md5(str(sorted_entities).encode('utf-8')).hexdigest()

# wiki id --> title
cache_path = './data/cache/ent_titles.{}.cache.pkl'.format(digest)
if os.path.exists(cache_path):
    logger.info('loaded cache from {}.'.format(cache_path))
    with open(cache_path, 'rb') as fin:
        id_to_title = pickle.load(fin)
else:
    logger.info('build cache to {}.'.format(cache_path))
    id_to_title = {}
    with open(wiki_ids_path, 'r', encoding='utf-8') as fin:
        for line in tqdm(fin):
            row = line.strip().split('\t')
            if row[0] in relateness_entities:
                id_to_title[row[0]] = row[1]
    with open(cache_path, 'wb') as fout:
        pickle.dump(id_to_title, fout)
# word util
word_utils = WordUtils()

# file reader
reader = HyberReader()

# tokenizer
sp = spacy.load('en_core_web_sm', disable=['ner', 'parser', 'targger'])
logger.info('successfully load all data.')

#############################################################################################
###
#                                      Functions

def compute_MAP(data, model, ent_utils):
    sum_avgp = 0.0
    num_queries = 0
    for k, v in data.items():
        _, e1 = k
        e1_index = torch.tensor([ent_utils.get_id_from_ent(e1)]).to(device)
        e1_vec = model.embedding(e1_index).cpu().numpy()[0]
        e2s = list(v.keys())
        e2_indexes = torch.tensor([ent_utils.get_id_from_ent(e2) for e2 in e2s]).to(device)
        e2_vecs = model.embedding(e2_indexes).cpu().numpy()
        sims = np.dot(e2_vecs, e1_vec.T)
        sorted_e2s = sorted(zip(e2s, sims), key=lambda x:x[1], reverse=True)
        num_correct = 0
        avgp = 0.0
        for i, (e2, _) in enumerate(sorted_e2s):
            if v[e2] == 1:
                num_correct += 1
                avgp += num_correct / (i+1)
        if num_correct > 0:
            avgp /= num_correct
            sum_avgp += avgp
            num_queries += 1
    return sum_avgp / num_queries

def spacy_tokenize(text):
    text = text.strip()
    if len(text) == 0:
        return []
    text = re.sub(r'\s+', ' ', text)
    tokens = sp(text)
    tokens = [token.text for token in tokens]
    return tokens

def get_batch(ent_utils):
    lines = reader.patch_of_lines(config.BATCH_SIZE)

    # set empty batch now
    ctxt_word_ids = []
    # ent_component_words = []
    # ent_wikiids = []
    ent_idxes = []
    targets = []

    for line in lines:
        row = line.split('\t')

        if len(row) == 2:
            ent_wikiid = row[0]
            line_words = row[1].split(' ')
        elif len(row) == 4:
            ent_wikiid = row[0]
            left_words = row[2].split(' ')
            right_words = row[3].split(' ')
            line_words = []
            line_words += left_words[max(0, len(left_words) - config.HYP_CTXT_LEN):]
            line_words += right_words[:min(len(right_words), config.HYP_CTXT_LEN)]
        else:
            logger.info("Invalid line: {}".format(row))
        
        positive_words = []
        for word in line_words:
            if word_utils.contains_word(word):
                positive_words.append(word)

        # Try getting some words from the entity title if the canonical page is empty.
        if len(positive_words) == 0:
            ent_title = id_to_title[ent_wikiid]
            for word in spacy_tokenize(ent_title):
                if word_utils.contains_word(word):
                    positive_words.append(word)

        # Still empty ? Get some random words then.
        if len(positive_words) == 0:
            positive_words.append(word_utils.get_word_from_id(word_utils.random_w_id()[0]))


        local_ent_idx = ent_utils.get_id_from_ent(ent_wikiid)
        local_targets = []
        local_ctxt_word_ids = word_utils.random_w_id(k = config.NUM_WORDS_PER_ENT* config.NUM_NEG_WORDS)

        
        for i in range(config.NUM_WORDS_PER_ENT):
            select_pos_w = random.choice(positive_words)
            select_pos_w_id = word_utils.get_id_from_word(select_pos_w)
            grd_trth = random.randint(0, config.NUM_NEG_WORDS-1)

            local_ctxt_word_ids[i*config.NUM_NEG_WORDS+grd_trth] = select_pos_w_id
            local_targets.append(grd_trth)
        

        ctxt_word_ids.append(local_ctxt_word_ids)
        ent_idxes.append(local_ent_idx)
        targets.append(local_targets)

    ctxt_word_vecs = word_utils.get_embeddings(torch.tensor(ctxt_word_ids).view(-1)).to(device)
    ent_idxes = torch.tensor(ent_idxes).to(device)
    targets = torch.tensor(targets).view(-1).to(device)

    return (ctxt_word_vecs, ent_idxes), targets


def train():
    cache_path = './data/cache/init_ent_info.{}.cache.pkl'.format(digest)
    if os.path.exists(cache_path):
         logger.info('loaded cache from {}.'.format(cache_path))
         with open(cache_path, 'rb') as fin:
             data = pickle.load(fin)
             ent_utils = data['ent_utils']
             ent_init_embeddings = data['ent_init_embeddings']
    else:
        logger.info('build cache to {}.'.format(cache_path))
        ent_utils = EntityUtils(sorted_entities)
        ent_init_embeddings = np.random.randn(ent_utils.get_total_ent_num(), config.EMBEDDING_SIZE)
        ent_init_embeddings[0] = np.zeros(config.EMBEDDING_SIZE)
        for ent in tqdm(sorted_entities, ascii=True):
            ent_title = id_to_title[ent]
            ent_id = ent_utils.get_id_from_ent(ent)
            ent_tokens = spacy_tokenize(ent_title)
            num_words_title = 0
            init_vec = np.zeros(config.EMBEDDING_SIZE)
            for token in ent_tokens:
                if word_utils.contains_word(token):
                    init_vec += word_utils.get_vec_from_word(token)
                    num_words_title += 1
            if num_words_title > 0:
                init_vec /= num_words_title
                ent_init_embeddings[ent_id] = init_vec
        data = {'ent_utils': ent_utils, 'ent_init_embeddings': ent_init_embeddings}
        with open(cache_path, 'wb') as fout:
            pickle.dump(data, fout)
    logger.info("check entity init embedding: {}, which should be -0.022287...".format(ent_init_embeddings[1][0]))

    model = EntToVecModel(ent_utils.get_total_ent_num(), torch.tensor(ent_init_embeddings).float()).to(device)
    optimizer = torch.optim.Adagrad(model.parameters(), lr=config.LEARNING_RATE)
    criterion = nn.MultiMarginLoss(margin=0.1)
    model.train()

    test_every_num_epochs = 1 
    save_every_num_epochs = 5

    logger.info('test begin trian:')
    with torch.no_grad():
        score = compute_MAP(relateness_valid_queries, model, ent_utils)
        logger.info("Score on valid dataset = {}".format(score))

    logger.info("begin training")
    for epoch in range(config.MAX_EPOCH):
        start_time = time.time()
        total_loss = 0.0

        for _ in tqdm(range(config.NUMBER_BATCHES_PER_EPOCH), desc="Epoch #{}".format(epoch), ascii=True):
            optimizer.zero_grad()
            x, y = get_batch(ent_utils)
            output = model(x)
            loss = criterion(output, y)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

        cost_time = time.time() - start_time

        if epoch % test_every_num_epochs == 0:
            with torch.no_grad():
                score = compute_MAP(relateness_valid_queries, model, ent_utils)
                logger.info("Score on valid dataset = {}, epoch num = {}.".format(score, epoch))

        if epoch % save_every_num_epochs == 0:
            model_path = './model/e2v_model.{}.{}.ckpt'.format(task, epoch)
            logger.info("Save model to: {}".format(model_path))
            torch.save(model.state_dict(), model_path)

        logger.info("Epoch {}: total loss = {}, time cost = {}s.".format(epoch, total_loss, cost_time))

def test():
    cache_path = './data/cache/init_ent_info.{}.cache.pkl'.format(digest)
    if os.path.exists(cache_path):
         logger.info('loaded cache from {}.'.format(cache_path))
         with open(cache_path, 'rb') as fin:
             data = pickle.load(fin)
             ent_utils = data['ent_utils']
             ent_init_embeddings = data['ent_init_embeddings']

    for epoch in range(60,121,5):
        ckpt_path = './model/e2v_model.{}.{}.ckpt'.format(task, epoch)
        model = EntToVecModel(ent_utils.get_total_ent_num())
        model.load_state_dict(torch.load(ckpt_path))
        model = model.to(device)
        model.eval()

        with torch.no_grad():
            score = compute_MAP(relateness_test_queries, model, ent_utils)
            print(epoch, ': ', score)


def main():
    # train()
    test()

if __name__ == '__main__':
    main()