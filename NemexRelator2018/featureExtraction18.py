import itertools
from parameters18 import *
import sys
import ast


def read_record(file):
    with open(file) as f:
        return [ast.literal_eval(line) for line in f]


def split_record(rec, test_idx, folds):
    record_train, record_test = [], []
    if folds == 0:  # SemEval's test set is used here
        for rec in record:
            if rec[0] in test_idx['1.1']:
                record_test.append(rec)
            else:
                record_train.append(rec)
        train_record_output = path_to_feat_folder + 'record_train.txt'
        test_record_output = path_to_feat_folder + 'record_test.txt'
        write_split(record_train, record_test, train_record_output, test_record_output)

    else:
        test_size = len(rec) / folds
        for k in range(1, folds+1):
            test_start = int(test_size * (k-1))
            test_end = int(test_size * k)
            record_test = rec[test_start:test_end]
            record_train = rec[:test_start] + rec[test_end:]
            train_record_output = path_to_feat_folder + 'record_train' + str(k) + '.txt'
            test_record_output = path_to_feat_folder + 'record_test' + str(k) + '.txt'
            write_split(record_train, record_test, train_record_output, test_record_output)

    return record_train, record_test


def write_split(record_train, record_test, train_out, test_out):
    with open(train_out, 'w+') as rec_train:
        for rec in record_train:
            rec_train.write(str(rec) + '\n')

    with open(test_out, 'w+') as rec_test:
        for rec in record_test:
            rec_test.write(str(rec) + '\n')


def index_vocab(voc_file, uniq_wrds):
    with open(voc_file, 'w+') as voc:
        for wrd in uniq_wrds:
            voc.write(wrd + '\n')


def index_shapes(shp_file):
    uniq_shps = list(itertools.product(range(2), repeat=7))  # repeat number should correspond to dims of shape vector
    with open(shp_file, 'w+') as shps:
        for shp in uniq_shps:
            shps.write(str(shp) + '\n')


def index_contexts(rec, e1_con_out, e2_con_out):
    # for now, just taking contexts if they're not entities
    uniq_e1_con, uniq_e2_con = set(), set()

    for r in rec:
        if len(r[1]) < 3:
            continue
        else:
            uniq_e1_con.add(r[1][1])
            uniq_e2_con.add(r[1][-2])

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
                idx[task] = {abs}
            else:
                idx[task].add(abs)

    return idx


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('featureExtraction18.py script requires 1 argument')
        exit(1)

    fold = int(sys.argv[1])
    record_file = path_to_feat_folder + 'record.txt'
    vocab_output = path_to_feat_folder + 'vocab.txt'
    shapes_output = path_to_feat_folder + 'shapes.txt'
    e1_context_output = path_to_feat_folder + 'e1_context.txt'
    e2_context_output = path_to_feat_folder + 'e2_context.txt'
    eval_file = 'training-eval.txt'
    testing_abs_index = create_testing_index(eval_file)
    record = read_record(record_file)
    unique_words = set([word for rec in record for word in rec[1]])
    train_record, test_record = split_record(record, testing_abs_index, fold)
    index_vocab(vocab_output, unique_words)
    index_shapes(shapes_output)
    index_contexts(record, e1_context_output, e2_context_output)
