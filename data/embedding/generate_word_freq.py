# coding=utf-8
from tqdm import tqdm

wiki_content_words_path = './data/wiki/wiki_content_words.txt'
wiki_word_freq_path = './data/wiki/word_freq.txt'

def main():
    freq = {}
    with open(wiki_content_words_path, 'r', encoding='utf-8') as fin:
        for line in tqdm(fin):
            row = line.strip().split('\t')
            words = row[1].split(' ')
            for word in words:
                if word not in freq:
                    freq[word] = 1
                else:
                    freq[word] += 1

    freq = {k: v for k, v in freq.items() if v >= 10}
    sorted_freq = sorted(freq.items(), key=lambda x:x[1], reverse=True)
    with open(wiki_word_freq_path, 'w', encoding='utf-8') as fout:
        for k, v in sorted_freq.items():
            fout.write('{}\t{}\n'.format(k, v))

if __name__ == '__main__':
    main()