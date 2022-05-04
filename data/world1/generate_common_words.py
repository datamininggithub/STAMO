# coding=utf-8
import os, pickle
from tqdm import tqdm
from gensim.models import KeyedVectors

dataset_vocab_path = './data/world1/vocab.pkl'
word2vec_path = './data/GoogleNews-vectors-negative300.bin'
stop_word_path = './data/stopwords6.txt'
cache_path = './data/cache/w1.common_word.cache.pkl'

def load_common_word():
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as fin:
            common_word = pickle.load(fin)
    else:
        # 移除stop word和数字
        stop_word = set()
        with open(stop_word_path, 'r', encoding='utf-8') as fin:
            for line in fin:
                stop_word.add(line.strip())
        def is_stop_word_or_number(w):
            if len(w) <= 1 or w.lower() in stop_word or w.isnumeric():
                return True
            else:
                return False

        '''
        word_freq = {}
        with open(wiki_word_freq_path, 'r', encoding='utf-8') as fin:
            for line in tqdm(fin):
                row = line.strip().split('\t')
                word = row[0]
                freq = int(row[1])
                if not is_stop_word_or_number(word):
                    word_freq[word] = freq
        '''
        with open(dataset_vocab_path, 'rb') as fin:
            vocab = pickle.load(fin)

        common_word = {}
        word2vec = KeyedVectors.load_word2vec_format(word2vec_path, binary=True)
        for word, wid in vocab.items():
            # if (not is_stop_word_or_number(word)) and (word in word2vec):
            if word in word2vec:
                common_word[word] = 1

        with open(cache_path, 'wb') as fout:
            pickle.dump(common_word, fout)

    return common_word

if __name__ == '__main__':
    common_word = load_common_word()
    print(type(common_word), len(common_word))