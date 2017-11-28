try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
import re


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
                    ent_text = el.text
                    ent_idx[ent_id] = ent_text  # collect id to entity mapping to be used later
                    ent_tail = el.tail
                    if ent_tail:
                        whole_abs_text += ent_id + ' ' + ent_tail
                    else:
                        whole_abs_text += ent_id
        txt_idx[abs_id] = whole_abs_text

    return txt_idx, ent_idx


def create_record(txts, rel_idx, ent_idx, rec_file):
    uniq_wrds, uniq_shps = set(), set()
    count = 0
    with open(rec_file, 'w+') as rec:
        for abs_id, rels in rel_idx.items():
            for rel, info in rels.items():
                count += 1
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
                to_write = tuple([count, tokens_with_punc, e1, e2, rel, s_len])
                rec.write(str(to_write) + '\n')

                for i, token in enumerate(tokens_with_punc):
                    shape_vec = [0, 0, 0, 0, 0, 0, 0]
                    if any(char.isupper() for char in token):
                        shape_vec[0] = 1
                    if '-' in token:
                        shape_vec[1] = 1
                    if any(char.isdigit() for char in token):
                        shape_vec[2] = 1
                    if i == 0 and token[0].isupper():
                        shape_vec[3] = 1
                    if token[0].islower():
                        shape_vec[4] = 1
                    if '_' in token:
                        shape_vec[5] = 1
                    if '"' in token:
                        shape_vec[6] = 1
                    uniq_shps.add(tuple(shape_vec))

    return uniq_wrds, uniq_shps


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


def index_shapes(shp_file, uniq_shps):
    with open(shp_file, 'w+') as shps:
        for shp in uniq_shps:
            shps.write(str(shp) + '\n')


if __name__ == '__main__':
    path_to_relations = '1.1.relations.txt'
    path_to_data = '1.1.text.xml'
    record_output = 'features/record.txt'
    vocab_output = 'features/vocab.txt'
    shapes_output = 'features/shapes.txt'
    relation_output = 'features/relations.txt'
    relation_index = create_relation_index(path_to_relations, relation_output)
    text_index, entity_index = collect_texts(path_to_data)
    unique_words, unique_shapes = create_record(text_index, relation_index, entity_index, record_output)
    index_vocab(vocab_output, unique_words)
    index_shapes(shapes_output, unique_shapes)