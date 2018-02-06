import ast
import sys
from parameters18 import *


def read_labels(file):
    labs = {}
    with open(file) as lbs:
        for i, lab in enumerate(lbs, start=1):
            labs[str(i)] = lab.strip()
    return labs


def read_record(file):
    with open(file) as f:
        return [ast.literal_eval(line) for line in f]


if __name__ == '__main__':
    which_set = sys.argv[1]
    submission = False
    if which_set == '0':
            which_set = ''
    elif which_set == "-1":
        which_set = ''
        submission = True
    path_to_predictions = path_to_model_folder + 'predictions.txt'
    path_to_labels = path_to_feat_folder + 'labels.txt'
    path_to_test = path_to_feat_folder + 'record_test' + which_set + '.txt'
    labels = read_labels(path_to_labels)
    test_record = read_record(path_to_test)
    entity_list = [(rec[2:4]) for rec in test_record]  # list of entities in order of test case
    with open('1.2.test.relations.txt') as test:
        submission_relations = test.readlines()

    if not submission:
        with open('answer_key18.txt', 'w+') as key:
            for rec in test_record:
                e1, e2, label = rec[2:5]
                label_split = label.split()
                if len(label_split) == 2:
                    key.write(label_split[0] + '(' + e1 + ',' + e2 + ',REVERSE)' + '\n')
                else:
                    key.write(label_split[0] + '(' + e1 + ',' + e2 + ')' + '\n')

    with open(path_to_predictions) as predicts, open(path_to_model_folder + 'predictions_with_labels.txt', 'w+') as out:
        for i, line in enumerate(predicts):
            e1, e2 = entity_list[i]
            label = labels[line.strip()]
            label_split = label.split()
            if not submission:
                if len(label_split) == 2:
                    out.write(label_split[0] + '(' + e1 + ',' + e2 + ',REVERSE)' + '\n')
                else:
                    out.write(label_split[0] + '(' + e1 + ',' + e2 + ')' + '\n')
            else:
                out.write(label_split[0] + submission_relations[i])