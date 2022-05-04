# coding=utf-8
from tqdm import tqdm
from gensim.models import KeyedVectors
import numpy as np
import hashlib, os, pickle
import torch
import torch.nn.functional as F
import random
from config import logger
from config import config
from embedding.generate_common_words import load_common_word as EMB_load_common_word
from world0.generate_common_words import load_common_word as W0_load_common_word
from world1.generate_common_words import load_common_word as W1_load_common_word
from worldIE.generate_common_words import load_common_word as IE_load_common_word
from worldIER.generate_common_words import load_common_word as IER_load_common_word

ds = {
    'EMB': EMB_load_common_word,
    'W0': W0_load_common_word,
    'W1': W1_load_common_word,
    'IE': IE_load_common_word,
    'IER': IER_load_common_word,
}

word2vec_path = './data/GoogleNews-vectors-negative300.bin'

class HyberReader:
    def __init__(self, total_num_passes_wiki_words=200):
        self.total_num_passes_wiki_words = total_num_passes_wiki_words
        self.num_passes_wiki_words = 0
        self.train_data_source = 'wiki-canonical'
        self.wiki_content_words_path = './data/wiki/wiki_content_final_words.IE.txt'
        self.wiki_hyper_words_path = './data/wiki/wiki_hyper_final_words.IE.txt'
        # self.wiki_content_words_path = './data/a.tmp'
        # self.wiki_hyper_words_path = './data/b.tmp'
        self.cur_path = self.wiki_content_words_path if self.num_passes_wiki_words < self.total_num_passes_wiki_words else self.wiki_hyper_words_path
        self.fin = open(self.cur_path, 'r', encoding='utf-8')

    def read_one_line(self):
        line = self.fin.readline()
        if len(line) == 0:
            # len(line)=0 --> EOF
            self.fin.close()
            self.num_passes_wiki_words += 1
            logger.info(self.num_passes_wiki_words)
            if self.num_passes_wiki_words == self.total_num_passes_wiki_words:
                print("switch to wiki hyperlinks now.")
                self.cur_path = self.wiki_hyper_words_path
            self.fin = open(self.cur_path, 'r', encoding='utf-8')
            line = self.fin.readline()
        return line

    def patch_of_lines(self, num):
        lines = []
        if num <= 0:
            return []
        cnt = 0
        while cnt < num:
            line = self.read_one_line().strip('\r\n')
            if len(line) > 0:
                lines.append(line)
                cnt += 1
        return lines

class EntityUtils:
    def __init__(self, entities):
        ent_to_idx = {}
        idx_to_ent = {}
        ent_to_idx['UNK_E'] = 0
        idx_to_ent[0] = 'UNK_E'
        total_ent_number = 1
        for ent in entities:
            ent_to_idx[ent] = total_ent_number
            idx_to_ent[total_ent_number] = ent
            total_ent_number += 1
        self.ent_to_idx = ent_to_idx
        self.idx_to_ent = idx_to_ent
        self.total_ent_number = total_ent_number

    def get_id_from_ent(self, ent):
        if ent in self.ent_to_idx:
            return self.ent_to_idx[ent]
        else:
            return 0

    def get_ent_from_id(self, e_id):
        if e_id in self.idx_to_ent:
            return self.idx_to_ent[e_id]
        else:
            return 'UNK_E'

    def get_total_ent_num(self):
        return self.total_ent_number

class WordUtils:
    def __init__(self, unig_power = 0.6, task=None):
        '''
        if config.task == 'EMB':
            from embedding.generate_common_words import load_common_word
        elif config.task == 'W0':
            from world0.generate_common_words import load_common_word
        elif config.task == 'W1':
            from world1.generate_common_words import load_common_word
        else:
            raise Exception('Error task: {}.'.format(config.task))
        '''
        if task is None:
            task = config.TASK

        self.word_to_idx = {}
        self.idx_to_word = {}
        self.unig_power = unig_power
        self.total_freq_at_unig_power = 0.0
        # self.w_f_at_unig_power_start = []
        # self.w_f_at_unig_power_end = []

        self.word_to_idx['UNK_W'] = 0
        self.idx_to_word[0] = 'UNK_W'
        self.word_to_idx['PAD_W'] = 1
        self.idx_to_word[1] = 'PAD_W'
        total_word_number = 2
        w_f_at_unig_power = [0.0, 0.0]

        word_freq = ds[task]()
        for word, freq in word_freq.items():
            self.word_to_idx[word] = total_word_number
            self.idx_to_word[total_word_number] = word
            total_word_number += 1

            if freq < 100:
                freq = 100
            freq_at_unig_power = freq**self.unig_power
            # self.total_freq_at_unig_power + freq_at_unig_power
            w_f_at_unig_power.append(freq_at_unig_power)
        w_f_at_unig_power = np.array(w_f_at_unig_power)
        w_f_at_unig_power = w_f_at_unig_power / np.sum(w_f_at_unig_power)
        self.total_word_number = total_word_number
        self.w_f_at_unig_power = w_f_at_unig_power

        self.MAX_Q_NUMBER = 300000
        self.random_cur_pos = 0
        self.random_pool = []

        digest = hashlib.md5(str(self.word_to_idx).encode('utf-8')).hexdigest()
        cache_path = './data/cache/w2v.{}.cache.pkl'.format(digest)

        if os.path.exists(cache_path):
            print('loaded cache from {}.'.format(cache_path))
            with open(cache_path, 'rb') as fin:
                self.M = pickle.load(fin)
        else:
            w2v = KeyedVectors.load_word2vec_format(word2vec_path, binary=True)
            w2v.init_sims(replace=True)
            print('loaded from {}, cache now.'.format(word2vec_path))
            M = np.zeros((len(self.word_to_idx), 300))
            found = 0
            for word in w2v.vocab:
                if word not in self.word_to_idx:
                    continue
                found += 1
                idx = self.word_to_idx[word]
                M[idx] = w2v[word]
            self.M = M
            with open(cache_path, 'wb') as fout:
                pickle.dump(M, fout)
        self.tensor_M = torch.tensor(self.M).float()
        print('check word embedding norm: {}'.format(np.sum(self.M[100]**2)))

    def random_w_id(self, k=1):
        # return np.random.choice(self.total_word_number, k, p=self.w_f_at_unig_power)
        if k + self.random_cur_pos > len(self.random_pool):
            new_pool = np.random.choice(self.total_word_number, self.MAX_Q_NUMBER, p=self.w_f_at_unig_power).tolist()
            self.random_pool = self.random_pool[self.random_cur_pos:] + new_pool
            self.random_cur_pos = 0
        ret = self.random_pool[self.random_cur_pos: k+self.random_cur_pos]
        self.random_cur_pos += k
        return ret

    def get_id_from_word(self, word):
        if word in self.word_to_idx:
            return self.word_to_idx[word]
        else:
            return 0

    def get_word_from_id(self, w_id):
        if w_id in self.idx_to_word:
            return self.idx_to_word[w_id]
        else:
            return 'UNK_W'

    def contains_word(self, word):
        if word in self.word_to_idx and self.word_to_idx[word] != 0:
            return True
        else:
            return False

    def get_vec_from_word(self, word):
        return self.get_vec_from_id(self.get_id_from_word(word))

    def get_vec_from_id(self, w_id):
        return self.M[w_id]

    def get_embeddings(self, input):
        return F.embedding(input, self.tensor_M)

def main():
    # word_util = WordUtils()

    # HyberReader unit test
    '''
    reader = HyberReader(total_num_passes_wiki_words=4)
    for _ in range(20):
        print(reader.patch_of_lines(20))
    '''


if __name__ == '__main__':
    main()