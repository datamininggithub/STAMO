# coding=utf-8
import os, re, spacy
from tqdm import tqdm
from WikiExtractor import findBalanced, replaceInternalLinks, replaceInternalLinksWithTracker
from collections import defaultdict
from multiprocessing import Process, Queue


begin_pattern = re.compile(r"<doc id=\"(\S+)\" url=\"(\S+)\" title=\"(.+)\" mark=\"(\S+)\">")
end_pattern = re.compile(r"</doc>")
redirect_pattern = re.compile(r"#REDIRECT\s+\[\[(.+)\]\]")

wiki_ids_path = './data/wiki/ids.txt'
wiki_redirect_to_id_path = './data/wiki/redirect_to_id.txt'
wiki_content_words_path = './data/wiki/wiki_content_words.txt'
wiki_hyper_words_path =  './data/wiki/wiki_hyper_words.txt'

# spacy.require_gpu()

def normalize_title(title, keep_section = False):
    origin_title = title
    title = title.replace('_', ' ').strip()
    title_item = title.split(' ')
    if len(title_item) == 0 or len(title_item[0]) == 0:
        return ''
    title_item[0] = title_item[0][0].upper() + title_item[0][1:]
    title = ' '.join(title_item)
    if not keep_section:
        pos = title.find('#')
        if pos > 0:
            title = title[:pos]
    return title

def normalize_mention(mention):
    # mention = _run_strip_accents(mention)
    mention = mention.strip()
    mention = re.sub(r'\s+', ' ', mention)
    return mention

def data_generate(i, q_in, q_content_out, q_hyper_out, title_to_id, redirect_to_id):
    
    state = 0
    doc_id = ""
    doc_title = ""
    doc_mark = ""
    doc_tokens = []
    mentions = []


    sp = spacy.load('en_core_web_sm', disable=['ner', 'parser', 'targger'])
    def spacy_tokenize(text):
        text = text.strip()
        if len(text) == 0:
            return []
        text = re.sub(r'\s+', ' ', text)
        tokens = sp(text)
        tokens = [token.text for token in tokens]
        return tokens

    while True:
        res = q_in.get()
        if res is None:
            break
        path = res
        total_line = 0
        with open(path, 'r', encoding='utf-8') as fin:
            for line in fin:
                total_line += 1
        with open(path, 'r', encoding='utf-8') as fin:
            for line in tqdm(fin, desc='{}'.format(path), position=i, total=total_line, bar_format="{l_bar}{r_bar}"):
                line = line.strip()
                if len(line) == 0:
                    continue
                match = begin_pattern.match(line)
                if match:
                    assert state == 0
                    state = 1
                    doc_id = match.group(1)
                    doc_title = match.group(3)
                    doc_mark = match.group(4)
                    doc_tokens.clear()
                    mentions.clear()
                    continue
                match = end_pattern.match(line)
                if match:
                    assert state == 1
                    state = 0
                    if doc_mark != 'article':
                        continue
                    # debug: 有一些文档内容为空
                    if len(doc_tokens) > 0:
                        q_content_out.put('{}\t{}\n'.format(doc_id, ' '.join(doc_tokens)))
                    for mention_start, mention_end, title, mention in mentions:
                        # todo: 考虑p(e|m)
                        title = normalize_title(title)
                        mention = re.sub(r'\s+', ' ', mention)
                        wiki_id = None
                        if title in title_to_id:
                            wiki_id = title_to_id[title]
                        elif title in redirect_to_id:
                            wiki_id = redirect_to_id[title]
                        if not wiki_id:
                            continue
                        left_tokens = doc_tokens[max(0, mention_start-30):mention_start]
                        right_tokens = doc_tokens[mention_end:min(mention_end+30, len(doc_tokens))]
                        if len(left_tokens) > 0 or len(right_tokens) > 0:
                            q_hyper_out.put('{}\t{}\t{}\t{}\n'.format(wiki_id, mention, ' '.join(left_tokens), ' '.join(right_tokens)))
                    continue
                if doc_mark == 'article':
                    line = re.sub(r"===(.*)===", r"\g<1>", line)
                    line = re.sub(r"==(.*)==", r"\g<1>", line)
                    text, mention_tracker = replaceInternalLinksWithTracker(line)
                    # print('*', text, mention_tracker)
                    if len(text) == 0:
                        continue
                    # tokens = sp(line)
                    # tokens = [token.text for token in tokens]
                    cur = 0
                    for track in mention_tracker:
                        mention_start, mention_end, mention_title = track[0], track[1], track[2]
                        tokens = spacy_tokenize(text[cur: mention_start])
                        doc_tokens += tokens
                        mention_tokens = spacy_tokenize(text[mention_start: mention_end])
                        mentions.append((len(doc_tokens), len(doc_tokens)+len(mention_tokens), mention_title, track[3]))
                        doc_tokens += mention_tokens
                        cur = mention_end
                    doc_tokens += spacy_tokenize(text[cur:])
                    # print('#', line_tokens, mentions)

def file_writer(path, q_out):
    fout = open(path, 'w', encoding='utf-8')
    while True:
        res = q_out.get()
        if res is None:
            break
        fout.write(res)
    fout.close()


def main():
    process_num = 24

    title_to_id = {}
    with open(wiki_ids_path, 'r', encoding='utf-8') as fin:
        for line in tqdm(fin):
            row = line.strip().split('\t')
            title_to_id[row[1]] = row[0]
    redirect_to_id = {}
    with open(wiki_redirect_to_id_path, 'r', encoding='utf-8') as fin:
        for line in tqdm(fin):
            row = line.strip().split('\t')
            if len(row) == 2:
                redirect_to_id[row[0]] = row[1]

    file_list = []
    for root, dirs, files in os.walk('./dump'):
        for file in files:
            path = root + "/" + file
            file_list.append(path)

    q_in = Queue()
    q_content_out = Queue()
    q_hyper_out = Queue()

    for path in tqdm(file_list[:], dynamic_ncols=True):
        q_in.put(path)

    generators = []
    for i in range(process_num):
        q_in.put(None)
        generator = Process(target=data_generate, args=(i, q_in, q_content_out, q_hyper_out, title_to_id, redirect_to_id))
        generator.start()
        generators.append(generator)

    content_writer = Process(target=file_writer, args=(wiki_content_words_path, q_content_out, ))
    hyper_writer = Process(target=file_writer, args=(wiki_hyper_words_path, q_hyper_out, )) 
    content_writer.start()
    hyper_writer.start()

    for generator in generators:
        generator.join()

    q_content_out.put(None)
    q_hyper_out.put(None)
    content_writer.join()
    hyper_writer.join()


if __name__ == '__main__':
    main()

