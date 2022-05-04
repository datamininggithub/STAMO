# coding=utf-8
import os, re, psycopg2, html, functools, unicodedata, pickle, math
from tqdm import tqdm
from WikiExtractor import findBalanced, replaceInternalLinks, replaceInternalLinksWithTracker
from collections import defaultdict


begin_pattern = re.compile(r"<doc id=\"(\S+)\" url=\"(\S+)\" title=\"(.+)\" mark=\"(\S+)\">")
end_pattern = re.compile(r"</doc>")
redirect_pattern = re.compile(r"#REDIRECT\s+\[\[(.+)\]\]")
link_pattern = re.compile(r"\[\[(.+)\]\]")

wiki_ids_path = './data/ids.txt'
wiki_redirect_to_title_path = './data/redirect_to_title.txt'
wiki_redirect_to_id_path = './data/redirect_to_id.txt'
wiki_df_path = './data/wiki_df.txt'
wiki_dict_path = './data/wiki_dict.txt'

wiki_pop_path = '.\\data\\wiki_pop.pkl'
wiki_wlm_path = '.\\data\\wiki_wlm.pkl'
sample_wiki_wlm_path = '.\\sample\\wiki_wlm.pkl'

wiki_content_words_path = './data/tmp_wiki_content_words.txt'
wiki_hyper_words_path =  './data/tmp_wiki_hyper_words.txt'

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


def add_count(mention, entity, counter, keeped_entity):
    if len(mention) == 0 or (keeped_entity and entity not in keeped_entity):
        return
    if mention not in counter:
        counter[mention] = {}
    mention_counter = counter[mention]
    if entity not in mention_counter:
        mention_counter[entity] = 1
    else:
        mention_counter[entity] += 1

def _run_strip_accents(text):
    """Strips accents from a piece of text."""
    text = unicodedata.normalize("NFD", text)
    output = []
    for char in text:
      cat = unicodedata.category(char)
      if cat == "Mn":
        continue
      output.append(char)
    return "".join(output)

def normalize_mention(mention):
    # mention = _run_strip_accents(mention)
    mention = mention.strip()
    mention = re.sub(r'\s+', ' ', mention)
    return mention


def get_disambig():
    # mention含有一些特殊字符(#, html tags等)时，也许应该做相应处理
    '''
    keeped_entity = set()
    with  open(wiki_df_path, 'r', encoding='utf-8') as fin:
        for line in tqdm(fin):
            row = line.strip().split('\t')
            if int(row[1]) >= 15:
                keeped_entity.add(row[0])
    '''
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
    for root, dirs, files in os.walk('./entity_dump'):
        for file in files:
            path = root + "/    " + file
            file_list.append(path)
    state = 0
    doc_id = ""
    doc_title = ""
    doc_mark = ""
    entities = []

    mention_candidates = {}
    local_add_count = functools.partial(add_count, counter=mention_candidates, keeped_entity=None)
    for title, wiki_id in tqdm(title_to_id.items()):
        mention = normalize_mention(title)
        local_add_count(mention, wiki_id)
    for title, wiki_id in tqdm(redirect_to_id.items()):
        mention = normalize_mention(title)
        local_add_count(mention, wiki_id)

    for file in tqdm(file_list):
        with open(file, 'r', encoding='utf-8') as fin:
            for line in tqdm(fin):
                match = begin_pattern.match(line)
                if match:
                    assert state == 0
                    state = 1
                    doc_id = match.group(1)
                    doc_title = match.group(3)
                    doc_mark = match.group(4)
                    entities.clear()
                    continue
                match = end_pattern.match(line)
                if match:
                    assert state == 1
                    state = 0
                    if len(entities) == 0:
                        continue
                    if doc_mark == 'disambig':
                        doc_title = re.sub(r"\(disambig\w*\)", "", doc_title).strip()
                        doc_mention = normalize_mention(doc_title)
                    for title, label in entities:
                        title = normalize_title(title)
                        mention = normalize_mention(label)
                        if title in title_to_id:
                            wiki_id = title_to_id[title]
                        elif title in redirect_to_id:
                            wiki_id = redirect_to_id[title]
                        else:
                            wiki_id = None
                        if wiki_id:
                            local_add_count(mention, wiki_id)
                            if doc_mark == 'disambig':
                                local_add_count(doc_mention, wiki_id)
                    continue
                match = link_pattern.match(line)
                if match:
                    inner = match.group(1)
                    pipe = inner.find('|')
                    if pipe < 0:
                        title = inner
                        label = title
                    else:
                        title = inner[:pipe].rstrip()
                        # find last |
                        curp = pipe + 1
                        for s1, e1 in findBalanced(inner):
                            last = inner.rfind('|', curp, s1)
                            if last >= 0:
                                pipe = last  # advance
                            curp = e1
                        label = inner[pipe + 1:].strip()
                    entities.append((title, label))


    with open(wiki_dict_path, 'w', encoding='utf-8') as fout:
        for mention, entity_count in tqdm(mention_candidates.items()):
            fout.write(mention)
            for entity, count in entity_count.items():
                fout.write('\t{}\t{}'.format(entity, count))
            fout.write('\n')

def space_tokenize(text):
    text = text.strip()
    if len(text) == 0:
        return []
    text = re.sub(r'\s+', ' ', text)
    return text.split(' ')

import spacy
sp = spacy.load('en_core_web_sm', disable=['ner', 'parser', 'targger'])

def spacy_tokenize(text):
    tokens = sp(text)
    tokens = [token.text for token in tokens]
    return tokens

def get_content_words():
    
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
    state = 0
    doc_id = ""
    doc_title = ""
    doc_mark = ""
    doc_tokens = []
    mentions = []

    content_fout = open(wiki_content_words_path, 'w', encoding='utf-8')
    hyper_fout = open(wiki_hyper_words_path, 'w', encoding='utf-8')

    for file in tqdm(file_list):
        with open(file, 'r', encoding='utf-8') as fin:
            for line in tqdm(fin):
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
                        content_fout.write('{}\t{}\n'.format(doc_id, ' '.join(doc_tokens)))
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
                        left_tokens = doc_tokens[max(0, mention_start-100):mention_start]
                        right_tokens = doc_tokens[mention_end:min(mention_end+100, len(doc_tokens))]
                        if len(left_tokens) > 0 or len(right_tokens) > 0:
                            hyper_fout.write('{}\t{}\t{}\t{}\n'.format(wiki_id, mention, ' '.join(left_tokens), ' '.join(right_tokens)))
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
    content_fout.close()
    hyper_fout.close()


def main():
    # get_ids()
    # get_redirect()
    # get_redirect_id()
    # get_wiki_df()
    # get_disambig()
    # get_features()
    get_content_words()
    

if __name__ == '__main__':
    main()

