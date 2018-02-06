import itertools
from parameters18 import *
import sys
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
import re


def create_relation_index(rel_path):
    rel_idx = {}
    typ_pattern = r'[^(]*'
    rel_pattern = r'\((.*?)\)'

    with open(rel_path) as f:
        for line in f:
            rel = re.findall(rel_pattern, line)[0]
            typ = re.findall(typ_pattern, line)[0]
            split_rel = rel.split(',')
            abs_id = split_rel[0].split('.')[0]
            if abs_id not in rel_idx:
                rel_idx[abs_id] = {}
            e1_e2 = tuple(split_rel[:2])
            len_split_rel = len(split_rel)
            if len_split_rel == 3:
                rel_idx[abs_id][e1_e2] = [typ, True]
            elif len_split_rel == 2:
                rel_idx[abs_id][e1_e2] = [typ, False]

    return rel_idx


def collect_texts(dat_file):
    txt_idx, ent_idx = {}, {}
    tree = ET.parse(dat_file)
    doc = tree.getroot()

    for txt in doc:  # looping over each abstract in entire xml doc
        abs_id = txt.get('id')
        whole_abs_text = ''
        for child in txt:  # children are title and abstract, H93-1076 has entities in title, but no relation
            for el in child.iter():
                tag = el.tag
                if tag == 'title':
                    continue
                elif tag == 'abstract':
                    abs_text = el.text
                    if abs_text:
                        whole_abs_text += abs_text
                elif tag == 'entity':
                    ent_id = el.get('id')
                    # ent_text = el.text
                    ent_text = ''.join(e for e in el.itertext() if e)
                    ent_idx[ent_id] = ent_text  # collect id to entity mapping to be used later
                    ent_tail = el.tail
                    if ent_tail:
                        if ent_tail[0] == ' ':
                            whole_abs_text += ent_id + ent_tail
                        else:
                            whole_abs_text += ent_id + ' ' + ent_tail
                    else:
                        whole_abs_text += ent_id + ' '
        txt_idx[abs_id] = whole_abs_text

    return txt_idx, ent_idx


def create_record(txts, rel_idx, ent_idx):
    recs = []

    for abs_id, rels in rel_idx.items():
        for rel, info in rels.items():
            e1, e2 = rel
            rel_patt = e1 + r'(.*?)' + e2
            rel_text_between = re.findall(rel_patt, txts[abs_id])[0]
            rel_text_full = e1 + rel_text_between + e2
            tokens = rel_text_full.split()
            # print(tokens)
            for i, token in enumerate(tokens):  # replace entity ids with actual entities
                if token in ent_idx:
                    if i == 0 or i == len(tokens)-1:  # if entity in relation, join with underscores if multi-word
                        tokens[i] = '_'.join(toke for toke in ent_idx[token].split())
                    else:
                        tokens[i] = ent_idx[token]
            tokens_with_punc = list(merge_punc(tokens))
            s_len = len(tokens_with_punc)
            typ = info[0] + ' REVERSE' if info[1] else info[0]
            recs.append(tuple([abs_id, tokens_with_punc, e1, e2, typ, s_len]))

    return recs


def write_record(rec_out, recs):
    with open(rec_out, 'w+') as out:
        for rec in recs:
            out.write(str(rec) + '\n')


def merge_punc(tkn_lst):
    to_merge = {',', '.', ':', ';'}
    seq = iter(tkn_lst)
    curr = next(seq)
    for nxt in seq:
        if nxt in to_merge:
            curr += nxt
        else:
            yield curr
            curr = nxt
    yield curr


def index_uniques(file, uniqs):
    with open(file, 'w+') as voc:
        for uniq in sorted(uniqs):
            voc.write(uniq + '\n')


def index_shapes(shp_file):
    uniq_shps = list(itertools.product(range(2), repeat=7))  # repeat number should correspond to dims of shape vector
    with open(shp_file, 'w+') as shps:
        for shp in uniq_shps:
            shps.write(str(shp) + '\n')


def index_contexts(uniq_e1_con, uniq_e2_con, e1_con_out, e2_con_out):
    # for now, just taking contexts if they're not entities
    with open(e1_con_out, 'w+') as e1_out:
        for u in uniq_e1_con:
            e1_out.write(u + '\n')
    with open(e2_con_out, 'w+') as e2_out:
        for u in uniq_e2_con:
            e2_out.write(u + '\n')


def create_testing_index(file):
    idx = {}
    with open(file) as f:
        for line in f:
            split = line.strip().split()
            if len(split) == 2:
                task, abs = split
            if task not in idx:
                idx[task] = [abs]
            else:
                idx[task].append(abs)

    return idx


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('featureExtraction18.py script requires one integer argument')
        exit(1)
    else:
        try:
            folds = int(sys.argv[1])
        except ValueError:
            print('argument for script featureExtraction18.py must be an integer')
            exit(1)

    path_to_relations = '1.2.relations.txt'
    path_to_test_relations = '1.2.test.relations.txt'
    path_to_data = '1.2.text.xml'
    path_to_test_data = '1.2.test.text.xml'
    train_record_file = path_to_feat_folder + 'record_train.txt'
    test_record_file = path_to_feat_folder + 'record_test.txt'
    vocab_output = path_to_feat_folder + 'vocab.txt'
    shapes_output = path_to_feat_folder + 'shapes.txt'
    e1_context_output = path_to_feat_folder + 'e1_context.txt'
    e2_context_output = path_to_feat_folder + 'e2_context.txt'
    label_output = path_to_feat_folder + 'labels.txt'
    eval_file = 'training-eval.txt'

    testing_abs_index = create_testing_index(eval_file)
    training_relation_index = create_relation_index(path_to_relations)
    test_relation_index = create_relation_index(path_to_test_relations)
    training_text_index, training_entity_index = collect_texts(path_to_data)
    test_text_index, test_entity_index = collect_texts(path_to_test_data)
    training_records = create_record(training_text_index, training_relation_index, training_entity_index)

    if folds == -1:
        test_records = create_record(test_text_index, test_relation_index, test_entity_index)
        write_record(train_record_file, training_records)
        write_record(test_record_file, test_records)
        unique_test_words = set([word for rec in test_records for word in rec[1]])
        unique_test_context_e1 = set([r[1][1] for r in test_records if len(r) >= 3])
        unique_test_context_e2 = set([r[1][-2] for r in test_records if len(r) >= 3])
    elif folds == 0:
        dev_training, dev_test = [], []
        for rec in training_records:
            if rec[0] in testing_abs_index['1.2']:  # change according to task
                dev_test.append(rec)
            else:
                dev_training.append(rec)
        write_record(train_record_file, dev_training)
        write_record(test_record_file, dev_test)
    else:
        test_size = len(training_records) / folds
        for k in range(1, folds + 1):
            test_start = int(test_size * (k-1))
            test_end = int(test_size * k)
            cv_test_split = training_records[test_start:test_end]
            cv_train_split = training_records[:test_start] + training_records[test_end:]
            cv_test_output = path_to_feat_folder + 'record_test' + str(k) + '.txt'
            cv_train_output = path_to_feat_folder + 'record_train' + str(k) + '.txt'
            write_record(cv_test_output, cv_test_split)
            write_record(cv_train_output, cv_train_split)

    unique_training_words = set([word for rec in training_records for word in rec[1]])
    unique_training_context_e1 = set([r[1][1] for r in training_records if len(r) >= 3])
    unique_training_context_e2 = set([r[1][-2] for r in training_records if len(r) >= 3])
    unique_words = unique_training_words.union(unique_test_words) if folds == -1 else unique_training_words
    unique_context_e1 = unique_training_context_e1.union(unique_test_context_e1) if folds == -1 else unique_training_context_e1
    unique_context_e2 = unique_training_context_e2.union(unique_test_context_e2) if folds == -1 else unique_training_context_e2
    unique_labels = {r[4] for r in training_records}
    index_uniques(vocab_output, unique_words)
    index_uniques(label_output, unique_labels)
    index_shapes(shapes_output)
    index_contexts(unique_context_e1, unique_context_e2, e1_context_output, e2_context_output)
