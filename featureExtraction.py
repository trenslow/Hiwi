import re
import itertools
import operator
import os
from parameters import *
import nltk


def clean_and_tokenize(sentence):
    e1_idx, e2_idx = 0, 0
    sentence = sentence[:-1] + ' ' + sentence[-1]
    e1_capture = re.findall(r'([^\s]*)<e1>(.*?)</e1>([^\s]*)', sentence)[0]
    e2_capture = re.findall(r'([^\s]*)<e2>(.*?)</e2>([^\s]*)', sentence)[0]
    if ' ' in e1_capture[1]:
        sentence = sentence.replace(e1_capture[1], '_'.join([word for word in e1_capture[1].split()]))
    if ' ' in e2_capture[1]:
        sentence = sentence.replace(e2_capture[1], '_'.join([word for word in e2_capture[1].split()]))
    tokens = sentence.split()
    for i, token in enumerate(tokens):
        if '<e1>' in token:
            e1_idx = i
            clean = token.replace('<e1>', '')
            clean = clean.replace('</e1>', '')
            tokens[i] = clean
        if '<e2>' in token:
            e2_idx = i
            clean = token.replace('<e2>', '')
            clean = clean.replace('</e2>', '')
            tokens[i] = clean

    # this block of code uses nltk to tokenize, and thus separates punctuation from certain words #
    # currently broken, as it breaks when creating vectors in sent2vec script #
    # dirty_tokens = nltk.word_tokenize(sentence)
    # remove = {'>', '<', 'e1', 'e2', '/e1', '/e2'}
    # for i, token in enumerate(dirty_tokens):
    #     if token == '>' and dirty_tokens[i-1] == '/e1':
    #         e1 = dirty_tokens[i-3]
    #     if token == '>' and dirty_tokens[i-1] == '/e2':
    #         e2 = dirty_tokens[i-3]
    # tokens = [token for token in dirty_tokens if token not in remove]
    # e1_idx = tokens.index(e1)
    # e2_idx = tokens.index(e2)
    return tokens, e1_idx + 1, e2_idx + 1  # plus 1 because counting starts from 1


def create_indexes(train_file, train):
    with open(train_file) as file:
        train1, train2 = itertools.tee(file, 2)
        sent_index = {}
        for line in itertools.islice(train1, 0, None, 4):
            split = line.strip().split('\t')
            index, sentence = int(split[0]), split[1][1:-1]
            sent_index[index] = sentence
        if train:
            lab_index = {int(index): label.strip() for index, label in
                         enumerate(itertools.islice(train2, 1, None, 4), start=1)}
        else:
            lab_index = {int(index): label.strip() for index, label in
                         enumerate(itertools.islice(train2, 1, None, 4), start=8001)}
    return sent_index, lab_index


def create_record(rec_file, sent_index, lab_index):
    with open(rec_file, 'w+') as record:
        for index, sentence in sorted(sent_index.items(), key=operator.itemgetter(0)):
            tokens, e1_index, e2_index = clean_and_tokenize(sentence)
            s_len = len(tokens)
            dist_e1_e2 = e2_index - (e1_index + 1)  # number of elements between entities
            dist_bos_e1 = e1_index - 1  # number of elements to left of e1
            dist_e2_eos = s_len - e2_index  # number of elements to right of e2
            label = lab_index[index]
            to_write = tuple([index, tokens, e1_index, e2_index, label, s_len, dist_e1_e2, dist_bos_e1, dist_e2_eos])
            record.write(str(to_write) + '\n')


def learn_labels(lab_file, lab_index):
    unique_labels = set(lab_index.values())
    with open(lab_file, 'w+') as labels:
        for label in sorted(unique_labels):
            labels.write(label + '\n')


def learn_words(wor_file, sent_index):
    unique_tokens = set()
    for sentence in sent_index.values():
        unique_tokens.update(clean_and_tokenize(sentence)[0])
    with open(wor_file, 'w+') as vocab:
        for token in unique_tokens:
            vocab.write(token + '\n')


def learn_suffixes(suf_file, sent_index):
    unique_suffixes = set()
    for sentence in sent_index.values():
        for token in clean_and_tokenize(sentence)[0]:
            if len(token) >= max_suffix_size:
                sliced_token = token[-max_suffix_size:]
            else:
                sliced_token = token
            unique_suffixes.update([token[i:] for i, char in enumerate(sliced_token)])
    with open(suf_file, 'w+') as suffixes:
        for suffix in unique_suffixes:
            suffixes.write(suffix + '\n')


def learn_shapes(shp_file, sent_index):
    unique_shapes = []
    for sentence in sent_index.values():
        tokens = clean_and_tokenize(sentence)[0]
        for i, token in enumerate(tokens):
            vector = [0, 0, 0, 0, 0, 0]
            if any(char.isupper() for char in token):
                vector[0] = 1
            if '-' in token:
                vector[1] = 1
            if any(char.isdigit() for char in token):
                vector[2] = 1
            if i == 0 and token[0].isupper():
                vector[3] = 1
            if token[0].islower():
                vector[4] = 1
            if '_' in token:
                vector[5] = 1
            tup_vec = tuple(vector)
            if tup_vec not in unique_shapes:
                unique_shapes.append(tup_vec)

    with open(shp_file, 'w+') as shapes:
        for shape in unique_shapes:
            shapes.write(str(shape) + '\n')


def learn_tags(tag_file):
    with open(tag_file, 'w+') as tags:
        for tag in nltk.data.load('help/tagsets/PY3/upenn_tagset.pickle').keys():
            tags.write(tag + '\n')


if __name__ == '__main__':
    if W < 0 or not isinstance(W, int):
        print('W must be a positive integer!')
        print('Please check the value of W in parameters.py script.')
        exit(1)
    if max_suffix_size < 0 or not isinstance(max_suffix_size, int):
        print('Max suffix size must be a positive integer!')
        print('Please check the value of max suffix size in parameters.py script.')
        exit(1)
    if after_e1 is False and before_e2 is False or after_e1 is True and before_e2 is True:
        print('Either after_e1 or before_e2 must be set to True (but not both) in parameters.py file.')
        exit(1)
    if not os.path.exists(path_to_feat_folder):
        os.makedirs(path_to_feat_folder)
    if not os.path.exists(path_to_model_folder):
        os.makedirs(path_to_model_folder)
    train_record_file = path_to_feat_folder + 'record_train.txt'
    test_record_file = path_to_feat_folder + 'record_test.txt'
    label_file = path_to_feat_folder + 'labels.txt'
    parameter_file = path_to_feat_folder + 'w_based_parameters.txt'
    words_file = path_to_feat_folder + 'vocab.txt'
    suffix_file = path_to_feat_folder + 'suffixes.txt'
    shape_file = path_to_feat_folder + 'shapes.txt'
    tags_file = path_to_feat_folder + 'tags.txt'
    train_sentence_index, train_label_index = create_indexes(path_to_train, True)
    test_sentence_index, test_label_index = create_indexes(path_to_test, False)
    print('writing training record file...')
    create_record(train_record_file, train_sentence_index, train_label_index)
    print('writing test record file...')
    create_record(test_record_file, test_sentence_index, test_label_index)
    print('writing label file...')
    learn_labels(label_file, train_label_index)
    print('writing words file...')
    learn_words(words_file, train_sentence_index)
    print('writing suffix file..')
    learn_suffixes(suffix_file, train_sentence_index)
    print('writing shape file...')
    learn_shapes(shape_file, train_sentence_index)
    print('writing POS tags file')
    learn_tags(tags_file)
