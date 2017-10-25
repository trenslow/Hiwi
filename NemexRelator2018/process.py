import xml.etree.cElementTree as ET
import re


def create_relation_index(rel_file):
    idx = {}
    typ_pattern = r'[^(]*'
    rel_pattern = r'\((.*?)\)'
    with open(rel_file) as f:
        for line in f:
            rel = re.findall(rel_pattern, line)[0]
            typ = re.findall(typ_pattern, line)[0]
            split_rel = rel.split(',')
            e1_e2 = tuple(split_rel[:2])
            len_split_rel = len(split_rel)
            if len_split_rel == 3:
                idx[e1_e2] = [typ, True]
            elif len_split_rel == 2:
                idx[e1_e2] = [typ, False]
    return idx


def collect_data(dat_file):
    idx = {}
    tree = ET.parse(dat_file)
    root = tree.getroot()
    for elem in root.iter():
        # print(elem.tag, elem.attrib)
        if elem.tag == 'text':
            abs_id = elem.attrib['id']
            idx[abs_id] = {}
        elif elem.tag == 'abstract':
            # abs_text = ''.join(elem.itertext())
            for yo in elem:
                print(yo.text, yo.tail)
            # idx[abs_id] = {i: sent for i, sent in enumerate(abs_text.split(' '))}

    return idx



if __name__ == '__main__':
    path_to_relations = 'SemEval2018_task7/1.1.relations.txt'
    path_to_data = 'SemEval2018_task7/1.1.text.xml'
    relation_index = create_relation_index(path_to_relations)
    abstract_index = collect_data(path_to_data)
    for k,v in abstract_index.items():
        for y,o in v.items():
            print(y,o)