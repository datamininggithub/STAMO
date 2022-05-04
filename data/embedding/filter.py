# coding=utf-8
import pickle, spacy
from tqdm import tqdm

relateness_data_path = './data/dataset/relateness.W1.pkl'
wiki_content_words_path = './data/wiki/wiki_content_words.txt'
wiki_hyper_words_path = './data/wiki/wiki_hyper_words.txt'
wiki_content_words_final_path = './data/wiki/wiki_content_final_words.W1.txt'
wiki_hyper_words_final_path = './data/wiki/wiki_hyper_final_words.W1.txt'

def main():
    with open(relateness_data_path, 'rb') as fin:
        data = pickle.load(fin)
    entities = data['entities']

    def filter(fin, fout):
        for line in tqdm(fin):
            row = line.strip().split('\t')
            if row[0] in entities:
                fout.write(line)

    with open(wiki_content_words_path, 'r') as fin, open(wiki_content_words_final_path, 'w') as fout:
        filter(fin, fout)
    with open(wiki_hyper_words_path, 'r') as fin, open(wiki_hyper_words_final_path, 'w') as fout:
        filter(fin, fout)

if __name__ == '__main__':
    main()
    # better_tokenize()