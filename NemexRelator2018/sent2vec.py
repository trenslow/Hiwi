import ast
from sklearn.model_selection import train_test_split


def read_record(file):
    with open(file) as f:
        return [ast.literal_eval(line) for line in f]


def read_shape_file(file, unkwn):
    shps = {}
    with open(file) as f:
        for i, line in enumerate(f):
            shps[ast.literal_eval(line)] = i
        shps[unkwn] = len(shps)
    return shps


def read_feat_file(file):
    feats = {}
    with open(file) as f:
        for i, item in enumerate(f):
            feats[item.strip()] = i
    return feats


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


if __name__ == '__main__':
    record_file = 'features/record.txt'
    vocab_file = 'features/vocab.txt'
    shapes_file = 'features/shapes.txt'
    relation_file = 'features/relation.txt'
    word_embds_file = '~/PycharmProjects/Hiwi/NemexRelator/2018/numberbatch-en.txt'
    unknown = 'UNK'
    records = read_record(record_file)
    record_train, record_test = train_test_split(records, test_size=0.1)
    vocab = read_feat_file(vocab_file)
    vocab[unknown] = len(vocab)
    shapes = read_shape_file(shapes_file, unknown)
    relations = read_feat_file(relation_file)