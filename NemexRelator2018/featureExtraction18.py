try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
import re
import itertools
from parameters18 import *
from sklearn.model_selection import train_test_split
import sys
import string



def create_relation_index(rel_path, rel_out):
    rel_idx = {}
    uniq_rels = set()
    typ_pattern = r'[^(]*'
    rel_pattern = r'\((.*?)\)'
    with open(rel_path) as f:
        for line in f:
            rel = re.findall(rel_pattern, line)[0]
            typ = re.findall(typ_pattern, line)[0]
            uniq_rels.add(typ)
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

    with open(rel_out, 'w+') as rel:
        for r in uniq_rels:
            rel.write(r + '\n')
            rel.write(r + ' REVERSE\n')
    return rel_idx


def collect_texts(dat_file):
    txt_idx, ent_idx = {}, {}
    tree = ET.parse(dat_file)
    doc = tree.getroot()

    # out = open('1.1.text.plain', 'w+')
    for txt in doc:  # looping over each abstract in entire xml doc
        abs_id = txt.get('id')
        whole_abs_text = ''
        # out_text = ''
        for child in txt:  # children are title and abstract, H93-1076 has entities in title, but no relation
            for el in child.iter():
                tag = el.tag
                if tag == 'title':
                    continue
                elif tag == 'abstract':
                    abs_text = el.text
                    if abs_text:
                        whole_abs_text += abs_text
                        # out_text += abs_text
                elif tag == 'entity':
                    ent_id = el.get('id')
                    ent_text = el.text
                    ent_idx[ent_id] = ent_text  # collect id to entity mapping to be used later
                    ent_tail = el.tail
                    if ent_tail:
                        if ent_tail[0] == ' ':
                            whole_abs_text += ent_id + ent_tail
                            # out_text += '_'.join(ent_text.split()) + ent_tail
                        else:
                            whole_abs_text += ent_id + ' ' + ent_tail
                            # out_text += '_'.join(ent_text.split()) + ' ' + ent_tail
                    else:
                        whole_abs_text += ent_id
                        # out_text += ent_id
        txt_idx[abs_id] = whole_abs_text
        # out.write(out_text + '\n')

    return txt_idx, ent_idx


def create_record(txts, rel_idx, ent_idx, test_idx, train_out, test_out, cross_val):
    uniq_wrds = set()
    records = []
    record_test, record_train = [], []

    for abs_id, rels in rel_idx.items():
        for rel, info in rels.items():
            e1, e2 = rel
            rel_patt = e1 + r'(.*?)' + e2
            rel_text_between = re.findall(rel_patt, txts[abs_id])[0]
            rel_text_full = e1 + rel_text_between + e2
            tokens = rel_text_full.split()
            for i, token in enumerate(tokens):  # replace entity ids with actual entities
                if token in ent_idx:
                    if i == 0 or i == len(tokens)-1:  # if entity in relation, join with underscores if multi-word
                        tokens[i] = '_'.join(toke for toke in ent_idx[token].split())
                    else:
                        tokens[i] = ent_idx[token]
            tokens_with_punc = list(merge_punc(tokens))
            s_len = len(tokens_with_punc)
            uniq_wrds.update(tokens_with_punc)
            rel = info[0] + ' REVERSE' if info[1] else info[0]
            records.append(tuple([abs_id, tokens_with_punc, e1, e2, rel, s_len]))

    if not cross_val:
        for rec in records:
            if rec[0] in test_idx['1.1']:  # code used for SemEval's test set in training phase
                record_test.append(rec)
            else:
                record_train.append(rec)
    else:
        record_train, record_test = train_test_split(records, test_size=0.12785)  # test size chosen to have same test instances as given by SemEval

    with open(train_out, 'w+') as rec_train:
        for rec in record_train:
            rec_train.write(str(rec) + '\n')

    with open(test_out, 'w+') as rec_test:
        for rec in record_test:
            rec_test.write(str(rec) + '\n')

    return uniq_wrds


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


def index_vocab(voc_file, uniq_wrds):
    with open(voc_file, 'w+') as voc:
        for wrd in uniq_wrds:
            voc.write(wrd + '\n')


def index_shapes(shp_file):
    uniq_shps = list(itertools.product(range(2), repeat=7))  # repeat number should correspond to dims of shape vector
    with open(shp_file, 'w+') as shps:
        for shp in uniq_shps:
            shps.write(str(shp) + '\n')


def create_testing_index(file):
    idx = {}
    with open(file) as f:
        for line in f:
            split = line.strip().split()
            if len(split) == 2:
                task, abs = split
            if task not in idx:
                idx[task] = {abs}
            else:
                idx[task].add(abs)

    return idx


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('featureExtraction18.py script requires 1 argument')
        exit(1)
    cross_validate = False if sys.argv[1] == 'false' else True
    path_to_relations = '1.1.relations.txt'
    path_to_data = '1.1.text.xml'
    train_record_output = path_to_feat_folder + 'record_train.txt'
    test_record_output = path_to_feat_folder + 'record_test.txt'
    vocab_output = path_to_feat_folder + 'vocab.txt'
    shapes_output = path_to_feat_folder + 'shapes.txt'
    relation_output = path_to_feat_folder + 'labels.txt'
    eval_file = 'training-eval.txt'
    relation_index = create_relation_index(path_to_relations, relation_output)
    text_index, entity_index = collect_texts(path_to_data)
    testing_abs_index = create_testing_index(eval_file)
    unique_words = create_record(text_index, relation_index, entity_index,
                                 testing_abs_index, train_record_output, test_record_output, cross_validate)
    index_vocab(vocab_output, unique_words)
    index_shapes(shapes_output)
