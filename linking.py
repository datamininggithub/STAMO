import random, pickle, argparse, time, sys, logging
import numpy as np
from config import config, logger, device, random_seed
from tqdm import tqdm
# from clueweb.dataset import *
from base_linker import *
from linker import ParamContextLinker
from world0.dataset import load_dataset as W0_load_dataset, load_vocab as W0_load_vocab
from world1.dataset import load_dataset as W1_load_dataset, load_vocab as W1_load_vocab
from worldIE.dataset import load_dataset as IE_load_dataset, load_vocab as IE_load_vocab
from worldIER.dataset import load_dataset as IER_load_dataset, load_vocab as IER_load_vocab
from functools import partial


random.seed(random_seed)
log_path = './log.txt'
anno_num = 20

# logger
ds = {
    'W0': [W0_load_dataset, W0_load_vocab],
    'W1': [W1_load_dataset, W1_load_vocab],
    'IE': [IE_load_dataset, IE_load_vocab],
    'IER': [IER_load_dataset, IER_load_vocab],
}


def test(dataset, vocab, linker, test_wiki=True):
    task = config.TASK
    wid_to_word = {v:k for k, v in vocab.items()}

    # Recal & Precision
    global_hit_count = 0
    global_precision_count = 0
    global_recall_count = 0
    # Accuracy
    global_acc_hit = 0
    global_acc_count = 0
    global_only_one = 0
    # OG Acc
    global_og_acc_hit = 0
    global_og_acc_count = 0
    start_time = time.time()
    cnt = 0

    for wiki_id, docs in dataset.items():
        if type(docs) == dict:
            if (not test_wiki) and (wiki_id == 'wiki'):
                continue
            logger.info('=== {} ==='.format(wiki_id))
            test(docs, vocab, linker)
            continue
        for doc in docs:
            for mention_index in doc.unknown_mentions:
                mention = doc.mentions[mention_index]
                digest = None
                candidates = mention.candidates
                logits = linker.disambiguation(mention, doc, wid_to_word, digest=digest)
                if len(candidates) == 1:
                    only_one += 1
                entity = candidates[np.argmax(logits)]
                mention.score = np.max(logits)
                for i, candidate in enumerate(candidates):
                    if candidate == wiki_id:
                        mention.score -= logits[i]
                        break
                mention.entity = entity
    has_test = False
    ans = {}
    for wiki_id, docs in dataset.items():
        if type(docs) == dict:
            continue
        has_test = True
        # for doc_index in query_entity.test_docs:
        hit_count = 0
        precision_count = 0
        recall_count = 0
        # Accuracy
        acc_hit = 0
        acc_count = 0
        only_one = 0
        # OG Acc
        og_acc_hit = 0
        og_acc_count = 0
        ans[wiki_id] = []
        for doc in docs:
            # doc = query_entity.docs[doc_index]
            for mention_index in doc.unknown_mentions:
                mention_info = doc.mentions[mention_index]
                if mention_info.entity != wiki_id:
                    ans[wiki_id].append((mention_info.entity, mention_info.golden_entity, mention_info.score))
                acc_count += 1
                if mention_info.entity == mention_info.golden_entity:
                    acc_hit += 1
                if mention_info.entity == wiki_id and mention_info.golden_entity == wiki_id:
                    hit_count += 1
                if mention_info.entity == wiki_id:
                    precision_count += 1
                if mention_info.golden_entity == wiki_id:
                    recall_count += 1

                if mention_info.golden_entity != wiki_id:
                    # print(query_entity.wiki_id, mention_info.golden_entity, mention_info.candidates, len(doc.unknown_mentions))
                    og_acc_count += 1
                    if mention_info.entity == mention_info.golden_entity:
                        og_acc_hit += 1
        if config.DETAIL_RESULT:
            acc = acc_hit / acc_count
            og_acc = og_acc_hit / og_acc_count if og_acc_count > 0 else 0.0
            precision = hit_count / precision_count if precision_count > 0 else 0.0
            recall = hit_count / recall_count
            fscore = 2*precision*recall/(precision+recall) if (precision+recall) > 0 else 0.0
            logger.info('Query: {}, Accuracy: {} ({}/{}, F-score: {})'.format(wiki_id, acc, acc_hit, acc_count, fscore))

        global_hit_count += hit_count
        global_precision_count += precision_count
        global_recall_count += recall_count
        # Accuracy
        global_acc_hit += acc_hit
        global_acc_count += acc_count
        global_only_one += only_one
        # OG Acc
        global_og_acc_hit += og_acc_hit
        global_og_acc_count += og_acc_count
    if has_test:
        cost_time = time.time() - start_time
        acc = global_acc_hit / global_acc_count
        og_acc = global_og_acc_hit / global_og_acc_count if global_og_acc_count > 0 else 0.0
        precision = global_hit_count / global_precision_count if global_precision_count > 0 else 0.0
        recall = global_hit_count / global_recall_count
        fscore = 2*precision*recall/(precision+recall) if (precision+recall) > 0 else 0.0
        f_acc = 2*og_acc*recall/(og_acc+recall) if og_acc>0 and recall>0 else 0.0
        logger.info('Accuracy: {} ({}/{}), Acc(OG): {} ({}/{}), Acc(F-Avg.): {}, Cost time: {}s'.format(acc, global_acc_hit, global_acc_count, og_acc, global_og_acc_hit, global_og_acc_count, f_acc, cost_time))
        logger.info('Precision: {} ({}/{}), Recall: {} ({}/{}), F-score: {}'.format(precision, global_hit_count, global_precision_count, recall, global_hit_count, global_recall_count, fscore))


def train(train_dataset, pool_dataset, valid_dataset, vocab, names, len_docs, linker):
    task = config.TASK
    print(len(train_dataset))
    wid_to_word = {v:k for k, v in vocab.items()}
    

    # Intra-Slot Optimization for SLOT 0
    for wiki_id, docs in tqdm(train_dataset.items()):
        for doc in docs[:500]:
            # doc = query_entity.docs[doc_index]
            for mention_index in doc.unknown_mentions:
                mention = doc.mentions[mention_index]
                linker.add_train_instance(mention, doc, wid_to_word)
    linker.train_feature()
    for epoch in range(450):
        linker.intra_slot_opt_one_epoch(epoch)


    # SLOT T
    for iter_num in range(1, 101):
        # assign pseudo labels:
        logger.info('#Train: {}, #Pool: {}'.format(anno_num, 1020-anno_num))
        logger.info('================== Iter #{} =================='.format(iter_num))
        logger.info('annotate pool dataset.')
        for wiki_id, docs in pool_dataset.items():
            for doc in docs:
                for mention_index in doc.unknown_mentions:
                    mention = doc.mentions[mention_index]
                    # digest = (doc.id, mention.mention, mention.golden_entity)
                    candidates = mention.candidates
                    logits = linker.disambiguation(mention, doc, wid_to_word, digest=None)
                    entity = candidates[np.argmax(logits)]
                    mention.entity = entity
                    mention.entity_list.append(entity)
                    mention.score = np.max(logits)
                    for i, candidate in enumerate(candidates):
                        if candidate == wiki_id:
                            mention.score -= logits[i]
                            break

        # update features of EEs
        logger.info('update features.')
        both_dataset = {}
        samples = {}
        for wiki_id in pool_dataset:
            both_dataset[wiki_id] = pool_dataset[wiki_id] + train_dataset[wiki_id]
            samples[wiki_id] = len_docs[wiki_id] / min(len_docs[wiki_id], len(both_dataset[wiki_id]))
        valid_func = None
        linker.update_feature(both_dataset, vocab, names, samples, valid_func=valid_func, max_epoch=35, lr=1e-2)
        # Intre-Slot Optimization
        linker.data.clear()
        for wiki_id, docs in tqdm(train_dataset.items()):
            for doc in docs[:500]:
                for mention_index in doc.unknown_mentions:
                    mention = doc.mentions[mention_index]
                    linker.add_train_instance(mention, doc, wid_to_word)
        linker.train_feature()
        for epoch in range(50):
            linker.intra_slot_opt_one_epoch(epoch)
        # Inter-Slot Optimization
        linker.inter_slot_opt()

        test(valid_dataset, vocab, linker)
    

def main():
    parser = argparse.ArgumentParser(description='Entity Linking')
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--unit", action="store_true")
    parser.add_argument("--cold", action="store_true")
    parser.add_argument("--detail_result", action="store_true")
    parser.add_argument("--task", default=None)
    args = parser.parse_args()

    task = args.task
    config.TASK = task
    config.UNIT_TEST = args.unit
    config.DETAIL_RESULT = args.detail_result
    config.COLD = args.cold

    if task == 'W1':
        train_dataset, test_dataset, names, all_candidates = ds[task][0]()
    else:
        train_dataset, test_dataset, names = ds[task][0]()
    vocab = ds[task][1]()

    anno_dataset = {}
    pool_dataset = {}
    samples = {}
    len_docs = {}
    if task == 'W1':
        logger.info('annotated doc #: {}'.format(anno_num))
        for wiki_id, docs in train_dataset.items():
            # labeled set
            anno_dataset[wiki_id] = []
            # unlabeled set
            pool_dataset[wiki_id] = []
            len_docs[wiki_id] = len(docs)
            samples[wiki_id] = len(docs) / min(len(docs), anno_num)
            for doc in docs[:anno_num]:
                for idx in doc.unknown_mentions:
                    mention = doc.mentions[idx]
                    mention.entity = mention.golden_entity
                    mention.entity_list.append(mention.golden_entity)
                doc.is_anno = True
                doc.golden_tokens = doc.tokens.copy()
                anno_dataset[wiki_id].append(doc)
            for doc in docs[anno_num:(1000+anno_num)]:
                doc.golden_tokens = doc.tokens.copy()
                pool_dataset[wiki_id].append(doc)

    if task == 'W1' and config.COLD:
        train_dataset = anno_dataset
        pool_dataset = pool_dataset
    else:
        train_dataset = train_dataset
        pool_dataset = None

    valid_func = partial(test, dataset=test_dataset, vocab=vocab, test_wiki=True)
    # saved_model = [None, False]
    saved_model = ['./model.bk/linking_model.IE.norm.400.ckpt', False]
    saved_model_path, saved_full_model = saved_model[0], saved_model[1]
    logger.info('load from: {}'.format(saved_model_path))
    linker = ParamContextLinker(names, all_candidates, saved_model_path=saved_model_path, saved_full_model=saved_full_model)

    if (not saved_full_model) and config.COLD:
        # SLOT 0
        logger.info('================== Iter #{} =================='.format(0))
        linker.update_feature(anno_dataset, vocab, names, samples, valid_func=None, max_epoch=600, lr=5e-2)


    if config.UNIT_TEST:
        logger.info("Unit test mode.")
    if args.train:
        train(train_dataset, pool_dataset, test_dataset, vocab, names, len_docs, linker)
    elif args.test:
        test(test_dataset, vocab, linker)
    else:
        raise Exception("Error args.")
    

if __name__ == '__main__':
    logger.info('run')
    main()
