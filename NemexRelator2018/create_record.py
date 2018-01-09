from parameters18 import *
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
            if r != 'COMPARE':
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


def create_record(txts, rel_idx, ent_idx):
    recs = []

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
            rel = info[0] + ' REVERSE' if info[1] else info[0]
            recs.append(tuple([abs_id, tokens_with_punc, e1, e2, rel, s_len]))

    return recs


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


if __name__ == '__main__':
    path_to_relations = '1.1.relations.txt'
    relation_output = path_to_feat_folder + 'labels.txt'
    relation_index = create_relation_index(path_to_relations, relation_output)
    path_to_data = '1.1.text.xml'
    text_index, entity_index = collect_texts(path_to_data)
    records = create_record(text_index, relation_index, entity_index)
    with open(path_to_feat_folder + 'record.txt', 'w+') as out:
        for rec in records:
            out.write(str(rec) + '\n')