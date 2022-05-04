# coding=utf-8
import pickle
from tqdm import tqdm

relateness_valid_path = './data/validate.svm'
relateness_test_path = './data/test.svm'
relateness_pkl_path = './data/relateness.IE.pkl'

extra_ents_path = './data/ents_in_all_docs.pkl'

wiki_ids_path = '.\\data\\ids.txt'
wiki_redirect_id_to_id_path = '.\\data\\redirect_id_to_id.txt'

def get_entity_list():
    id_to_title = {}
    with open(wiki_ids_path, 'r', encoding='utf-8') as fin:
        for line in tqdm(fin):
            row = line.strip().split('\t')
            id_to_title[row[0]] = row[1]
    redirect_id_to_id = {}
    with open(wiki_redirect_id_to_id_path, 'r', encoding='utf-8') as fin:
        for line in tqdm(fin):
            row = line.strip().split('\t')
            redirect_id_to_id[row[0]] = row[1]
            assert row[1] in id_to_title, row

    valid_queries = {}
    test_queries = {}
    entities = set()

    invalid_entity = set()
    def valid_entity(entity):
        if entity in id_to_title:
            return entity
        elif entity in redirect_id_to_id:
            return redirect_id_to_id[entity]
        else:
            if entity not in invalid_entity:
                print("Invalid entity: {}".format(entity))
                invalid_entity.add(entity)
            return None


    with open(relateness_valid_path, 'r', encoding='utf-8') as fin:
        for line in fin:
            row = line.strip().split(' ')
            label = int(row[0])
            qid = int(row[1].split(':')[1])
            ents = row[-2].split('-')
            e1 = valid_entity(ents[0])
            e2 = valid_entity(ents[1])

            if not (e1 and e2):
                continue

            entities.add(e1)
            entities.add(e2)

            if (qid, e1) not in valid_queries:
                valid_queries[(qid, e1)] = {}
            valid_queries[(qid, e1)][e2] = label

    with open(relateness_test_path, 'r', encoding='utf-8') as fin:
        for line in fin:
            row = line.strip().split(' ')
            label = int(row[0])
            qid = int(row[1].split(':')[1])
            ents = row[-2].split('-')
            e1 = valid_entity(ents[0])
            e2 = valid_entity(ents[1])

            if not (e1 and e2):
                continue

            entities.add(e1)
            entities.add(e2)

            if (qid, e1) not in test_queries:
                test_queries[(qid, e1)] = {}
            test_queries[(qid, e1)][e2] = label

    # add extrac ents
    with open(extra_ents_path, 'rb') as fin:
        extra_ents = pickle.load(fin)
    for ent in extra_ents:
        ent = valid_entity(ent)
        if ent:
            entities.add(ent)

    print(len(valid_queries), len(test_queries), len(entities))
    data = {'valid_queries': valid_queries, 'test_queries': test_queries, 'entities': entities}
    with open(relateness_pkl_path, 'wb') as fout:
        pickle.dump(data, fout)

def main():
    get_entity_list()

if __name__ == '__main__':
    main()