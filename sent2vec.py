import ast
import re
from parameters import *
from collections import Counter
import operator
import string
from nltk.tag import StanfordPOSTagger
import time


def read_record(file):
    with open(file) as f:
        return [ast.literal_eval(line) for line in f]


def read_tags(file):
    tags = {}
    with open(file) as f:
        for i, tag in enumerate(f):
            tags[tag.strip()] = i
    return tags


def read_shape_file(file, unkwn):
    shps = {}
    with open(file) as f:
        for i, line in enumerate(f):
            shps[ast.literal_eval(line)] = i
        shps[unkwn] = len(shps)
    return shps


def read_feat_file(file, unkwn):
    feats = {}
    with open(file) as f:
        for i, item in enumerate(f):
            feats[item.strip()] = i
        feats[unkwn] = len(feats)
    return feats


def read_param_file(file):
    params = {}
    with open(file) as f:
        for line in f:
            split = line.strip().split()
            params[split[0]] = int(split[1])
    return params


def read_clusters(file):
    clusts = {}
    with open(file) as f:
        for line in f:
            wrd, clust = line.strip().split()
            if 'marlin' in file:
                clusts[wrd] = int(clust)
            elif 'brown' in file:
                clusts[wrd] = int(clust, 2)
    if 'brown' in file:
        clusts['<RARE>'] = max(clusts.values()) + 1
    return clusts


def read_embeddings(file):
    embs = {}
    with open(file) as f:
        for line in f:
            split = line.strip().split()
            if len(split) == 2:
                continue
            else:
                word, vec = split[0], [float(val) for val in split[1:]]
                embs[word] = vec
    return embs

def create_pos_index(s_and_i, M, avg_m):
    posits = {}
    K = 2 * W + M
    vec_len = 2 * K + 2 + 1  # plus 2 for entities and plus 1 for middle
    for s, e1, e2 in s_and_i:
        norm_sent = normalize(M, [s, e1, e2], avg_m)
        vectors = vectorize(norm_sent, e1, e2, vec_len)
        for vector in vectors:
            if vector not in posits:
                posits[vector] = len(posits)
    posits['UNKNOWN'] = len(posits)
    return posits


def normalize(m, sent, avg_m):
    tkns, e1_index, e2_index = sent[0], sent[1], sent[2]
    l = e1_index - 1
    r = len(tkns) - e2_index
    current_m = e2_index - (e1_index + 1)
    if l == W and r == W:
        # print('1')
        if current_m > avg_m and use_avg_M:
            tkns = slice_middle(tkns, avg_m, e1_index, e2_index)
        else:
            tkns = pad_middle(tkns, m, e1_index, e2_index)
    elif l == W and r < W:
        # print('2')
        tkns = pad_right(tkns, r)
        if current_m > avg_m and use_avg_M:
            tkns = slice_middle(tkns, avg_m, e1_index, e2_index)
        else:
            tkns = pad_middle(tkns, m, e1_index, e2_index)
    elif l == W and r > W:
        # print('3')
        tkns = slice_right(tkns, r)
        if current_m > avg_m and use_avg_M:
            tkns = slice_middle(tkns, avg_m, e1_index, e2_index)
        else:
            tkns = pad_middle(tkns, m, e1_index, e2_index)
    elif r == W and l < W:
        # print('4')
        tkns = pad_left(tkns, l)
        if current_m > avg_m and use_avg_M:
            tkns = slice_middle(tkns, avg_m, e1_index, e2_index)
        else:
            tkns = pad_middle(tkns, m, e1_index, e2_index)
    elif r == W and l > W:
        # print('5')
        tkns = slice_left(tkns, l)
        if current_m > avg_m and use_avg_M:
            tkns = slice_middle(tkns, avg_m, e1_index, e2_index)
        else:
            tkns = pad_middle(tkns, m, e1_index, e2_index)
    elif l < W and r < W:
        # print('6')
        tkns = pad_left(tkns, l)
        tkns = pad_right(tkns, r)
        if current_m > avg_m and use_avg_M:
            tkns = slice_middle(tkns, avg_m, e1_index, e2_index)
        else:
            tkns = pad_middle(tkns, m, e1_index, e2_index)
    elif l > W and r > W:
        # print('7')
        tkns = slice_left(tkns, l)
        tkns = slice_right(tkns, r)
        if current_m > avg_m and use_avg_M:
            tkns = slice_middle(tkns, avg_m, e1_index, e2_index)
        else:
            tkns = pad_middle(tkns, m, e1_index, e2_index)
    elif l < W < r:
        # print('8')
        tkns = pad_left(tkns, l)
        tkns = slice_right(tkns, r)
        if current_m > avg_m and use_avg_M:
            tkns = slice_middle(tkns, avg_m, e1_index, e2_index)
        else:
            tkns = pad_middle(tkns, m, e1_index, e2_index)
    elif r < W < l:
        # print('9')
        tkns = slice_left(tkns, l)
        tkns = pad_right(tkns, r)
        if current_m > avg_m and use_avg_M:
            tkns = slice_middle(tkns, avg_m, e1_index, e2_index)
        else:
            tkns = pad_middle(tkns, m, e1_index, e2_index)
    return tkns


def vectorize(tokes, e1, e2, vec_length):
    vecs = []
    middle = vec_length // 2
    for i in range(1, len(tokes) + 1):
        vec = [0 for _ in range(vec_length)]
        val1 = i - e1
        val2 = i - e2
        vec[middle + val1] = 1
        vec[middle + val2] = 1
        if vec not in vecs:
            vecs.append(vec)

    return [tuple(v) for v in vecs]


def pad_middle(lst, m, e1, e2):
    for i in range(m - (e2 - e1) + 1):
        if after_e1:
            lst.insert(W+1, None)
        elif before_e2:
            lst.insert(len(lst)-(W+1), None)
    return lst


def slice_middle(lst, avg_m, e1, e2):
    if after_e1:
        return lst[:e1+avg_m] + lst[e2:]
    elif before_e2:
        return lst[:e1] + lst[e2-avg_m:]


def pad_left(lst, l):
    for i in range(W - l):
        lst.insert(0, None)
    return lst


def pad_right(lst, r):
    for i in range(W - r):
        lst.append(None)
    return lst


def slice_left(lst, l):
    return lst[l-W:]


def slice_right(lst, r):
    return lst[:-(r - W)]


if __name__ == '__main__':
    unknown = 'UNKNOWN'
    records_and_outs = [(path_to_feat_folder + 'record_train.txt', path_to_model_folder + 'libLinearInput_train.txt'),
                        (path_to_feat_folder + 'record_test.txt', path_to_model_folder + 'libLinearInput_test.txt')]
    path_to_tagger = 'stanford-postagger-2017-06-09/stanford-postagger-3.8.0.jar'
    path_to_model = 'stanford-postagger-2017-06-09/models/english-bidirectional-distsim.tagger'
    tagger = StanfordPOSTagger(path_to_model, path_to_tagger)
    if fire_embeddings:
        embeddings = read_embeddings(path_to_feat_folder + 'numberbatch-en.txt')
    if marlin:
        clusters = read_clusters(path_to_feat_folder + 'en_marlin_cluster_1000')
    elif brown:
        clusters = read_clusters(path_to_feat_folder + 'en_brown_1000')
    if fire_suffixes:
        suffixes = read_feat_file(path_to_feat_folder + 'suffixes.txt', unknown)
    if fire_shapes:
        shapes = read_shape_file(path_to_feat_folder + 'shapes.txt', unknown)
    if fire_tagger:
        tags = read_tags(path_to_feat_folder + 'tags.txt')

    for record_file, out_file in records_and_outs:
        which = re.findall(r'record_(.*?).txt', record_file)[0]
        print('creating LibLinear ' + which + ' file...')
        records = read_record(record_file)
        sentences_and_indexes = [(record[1], int(record[2]), int(record[3])) for record in records]
        sentence_labels = [record[4] for record in records]
        with open(path_to_feat_folder + 'labels.txt') as labs:
            labels = {lab.strip(): i for i, lab in enumerate(labs)}
        words = read_feat_file(path_to_feat_folder + 'vocab.txt', unknown)
        num_words = len(words) if fire_words else 0

        if 'train' in record_file:
            all_M_vals = [record[6] for record in records]
            M = max(all_M_vals)
            avg_M = sum(all_M_vals) // len(records)
            M_counts = Counter(all_M_vals)
            mode_M = max(M_counts.items(), key=operator.itemgetter(1))[0]
            if use_avg_M_plus_mode:
                avg_M += mode_M
            positions = create_pos_index(sentences_and_indexes, M, avg_M)
            num_positions = len(positions) if fire_positions else 0
        norm_sents = [normalize(M, sentence, avg_M) for sentence in sentences_and_indexes]
        if fire_tagger:
            tagged_sents = tagger.tag_sents([[w if w is not None else 'none' for w in sent] for sent in norm_sents])

        num_clusters = max(clusters.values()) + 1 if fire_clusters else 0
        num_embeddings = len(embeddings['0']) if fire_embeddings else 0
        num_suffixes = len(suffixes) if fire_suffixes else 0
        num_shapes = len(shapes) if fire_shapes else 0
        num_tags = len(tags) if fire_tagger else 0
        len_token_vec = num_words + num_positions + num_clusters + num_suffixes + num_shapes + num_tags + num_embeddings
        feat_val = ':1.0'

        with open(out_file, 'w+') as lib_out:
            in_clusters = 0
            out_clusters = 0
            for i, sentence in enumerate(sentences_and_indexes):
                sentence_feats = []
                current_label = sentence_labels[i]
                norm_sent = norm_sents[i]
                if fire_tagger:
                    tagged_sent = tagged_sents[i]
                K = 2 * W + M
                pos_vecs = vectorize(norm_sent, sentence[1], sentence[2], 2 * K + 2 + 1)
                for idx, token in enumerate(norm_sent):
                    offset = idx * len_token_vec
                    token_feats = []
                    if token:
                        if fire_words:
                            if token in words:
                                feat_pos = offset + words[token] + 1
                                token_feat = str(feat_pos) + feat_val
                                token_feats.append(token_feat)
                            else:
                                feat_pos = offset + words[unknown] + 1
                                token_feat = str(feat_pos) + feat_val
                                token_feats.append(token_feat)
                        if fire_positions:
                            if pos_vecs[idx] in positions:
                                feat_pos = offset + positions[pos_vecs[idx]] + num_words + 1
                                token_feat = str(feat_pos) + feat_val
                                token_feats.append(token_feat)
                            else:
                                feat_pos = offset + positions[unknown] + num_words + 1
                                token_feat = str(feat_pos) + feat_val
                                token_feats.append(offset + positions[unknown] + num_words + 1)
                        if fire_clusters:
                            temp_token = ''.join('0' if char.isdigit() else char for char in token)
                            if any(char.isalpha() for char in temp_token) and len(temp_token) > 1:
                                temp_token = ''.join(char for char in temp_token if char not in string.punctuation)
                            if temp_token in clusters:
                                feat_pos = offset + clusters[temp_token] + num_words + num_positions + 1
                                token_feat = str(feat_pos) + feat_val
                                token_feats.append(token_feat)
                            else:
                                feat_pos = offset + clusters['<RARE>'] + num_words + num_positions + 1
                                token_feat = str(feat_pos) + feat_val
                                token_feats.append(token_feat)
                        if fire_suffixes:
                            suffix_vec = []
                            for j in range(len(token)):
                                suffix = token[j:]
                                if suffix in suffixes:
                                    suffix_vec.append(suffixes[suffix])
                                else:
                                    if suffixes[unknown] not in suffix_vec:
                                        suffix_vec.append(suffixes[unknown])
                            for s in sorted(suffix_vec):
                                feat_pos = offset + s + num_words + num_positions + num_clusters + 1
                                token_feat = str(feat_pos) + feat_val
                                token_feats.append(token_feat)
                        if fire_shapes:
                            shape_vec = [0, 0, 0, 0, 0, 0]
                            if any(char.isupper() for char in token):
                                shape_vec[0] = 1
                            if '-' in token:
                                shape_vec[1] = 1
                            if any(char.isdigit() for char in token):
                                shape_vec[2] = 1
                            if idx == 0 and token[0].isupper():
                                shape_vec[3] = 1
                            if token[0].islower():
                                shape_vec[4] = 1
                            if '_' in token:
                                shape_vec[5] = 1
                            tup_vec = tuple(shape_vec)
                            if tup_vec in shapes:
                                feat_pos = offset + shapes[tup_vec] + num_words + num_positions + num_clusters + num_suffixes + 1
                                token_feat = str(feat_pos) + feat_val
                                token_feats.append(token_feat)
                            else:
                                feat_pos = offset + shapes[unknown] + num_words + num_positions + num_clusters + num_suffixes + 1
                                token_feat = str(feat_pos) + feat_val
                                token_feats.append(token_feat)
                        if fire_tagger:
                            current_tag = tagged_sent[idx][1]
                            feat_pos = offset + tags[current_tag] + num_shapes + num_words + num_positions + num_clusters + num_suffixes + 1
                            token_feat = str(feat_pos) + feat_val
                            token_feats.append(token_feat)
                        if fire_embeddings:
                            temp_token = ''.join('0' if char.isdigit() else char for char in token)
                            if any(char.isalpha() for char in temp_token) and len(temp_token) > 1:
                                temp_token = ''.join(char for char in temp_token if char not in string.punctuation)
                            if temp_token in embeddings:
                                vec = embeddings[temp_token]
                                token_feats += [str(offset + n + num_tags + num_shapes + num_words + num_positions +
                                                num_clusters + num_suffixes + 1) + ':' + str(vec[n])
                                                for n in range(num_embeddings)]

                    else:
                        if fire_words:
                            feat_pos = offset + words[unknown] + 1
                            unknown_word = str(feat_pos) + feat_val
                        else:
                            unknown_word = None
                        if fire_positions:
                            feat_pos = offset + positions[unknown] + num_words + 1
                            unknown_position = str(feat_pos) + feat_val
                        else:
                            unknown_position = None
                        if fire_clusters:
                            feat_pos = offset + clusters['<RARE>'] + num_words + num_positions + 1
                            unknown_cluster = str(feat_pos) + feat_val
                        else:
                            unknown_cluster = None
                        if fire_suffixes:
                            feat_pos = offset + suffixes[unknown] + num_words + num_positions + num_clusters + 1
                            unknown_suffix = str(feat_pos) + feat_val
                        else:
                            unknown_suffix = None
                        if fire_shapes:
                            feat_pos = offset + shapes[unknown] + num_words + num_positions + num_clusters + num_suffixes + 1
                            unknown_shape = str(feat_pos) + feat_val
                        else:
                            unknown_shape = None
                        if fire_tagger:
                            feat_pos = offset + tags['NN'] + num_shapes + num_words + num_positions + num_clusters + num_suffixes + 1
                            unknown_tag = str(feat_pos) + feat_val
                        else:
                            unknown_tag = None
                        token_feats = [unknown_word, unknown_position, unknown_cluster,
                                       unknown_suffix, unknown_shape, unknown_tag]

                    sentence_feats += token_feats

                lib_out.write(str(labels[current_label]) + ' ')
                lib_out.write(' '.join(i for i in sentence_feats if i) + '\n')