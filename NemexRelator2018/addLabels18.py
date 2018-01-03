import ast
import sys
from parameters18 import *


def read_labels(file):
    labs = {}
    with open(file) as lbs:
        for i, lab in enumerate(lbs):
            labs[str(i)] = lab.strip()
    return labs


if __name__ == '__main__':
    s, c, e = [''.join(char for char in arg if char != '.') for arg in sys.argv[1:]]
    lib_linear_params = 's' + s + 'c' + c + 'e' + e
    path_to_predictions = path_to_model_folder + lib_linear_params + '_predictions.txt'
    path_to_labels = path_to_feat_folder + 'labels.txt'
    path_to_test = path_to_feat_folder + 'record_test.txt'
    labels = read_labels(path_to_labels)

    test_map = {}  # for mapping test instances to their corresponding entity pair
    with open(path_to_test) as test, open('answer_key18.txt', 'w+') as key:
        for i, line in enumerate(test):
            rec = ast.literal_eval(line)
            e1, e2, lab = rec[2:5]
            # e1_id, e2_id = int(e1.split('.')[-1]), int(e2.split('.')[-1])
            # assert e1_id < e2_id
            test_map[i] = (e1, e2)
            lab_split = lab.split()
            if len(lab_split) == 2:
                key.write(lab_split[0] + '(' + e1 + ',' + e2 + ',REVERSE)' + '\n')
            else:
                key.write(lab_split[0] + '(' + e1 + ',' + e2 + ')' + '\n')

    with open(path_to_predictions) as predicts, open(path_to_model_folder + lib_linear_params + '_predictions_with_labels.txt', 'w+') as out:
        # out.write('1.1' + '\n')  # to specify which subtask we're participating in
        for i, line in enumerate(predicts):
            e1, e2 = test_map[i]
            label = labels[line.strip()]
            label_split = label.split()
            if len(label_split) == 2:
                out.write(label_split[0] + '(' + e1 + ',' + e2 + ',REVERSE)' + '\n')
            else:
                out.write(label_split[0] + '(' + e1 + ',' + e2 + ')' + '\n')
