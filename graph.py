from neo4j.v1 import GraphDatabase
import string
import sys


def read_file(file):
    rels = []

    with open(file) as f:
        for line in f:
            split = line.strip().split('\t')
            if len(split) > 1:
                extraction = tuple([arg.replace(' , ', ' ')[1:-1] for arg in split[1:-1]])
                rels.append(extraction)
    return rels


def clean_punctuation(text):
    return ''.join([char for char in text if char not in string.punctuation])


if __name__ == '__main__':
    extraction_file = sys.argv[1]  # need to do error catch here
    relations = read_file(extraction_file)
    uri = 'bolt://localhost:7687'
    driver = GraphDatabase.driver(uri, auth=('neo4j', 'password'))

    with driver.session() as sess:
        # clear out database (for development)

        sess.run("MATCH (n) DETACH DELETE n")

        for r in relations:
            arg1 = clean_punctuation(r[0])
            relation = clean_punctuation(r[1])
            arg2 = clean_punctuation(r[2])

            # create node for each arg. if exists, does nothing due to MERGE. need to adapt for n-ary relations later
            sess.run("MERGE (:arg1 {text:$arg1})", arg1=arg1)
            sess.run("MERGE (:arg2 {text:$arg2})", arg2=arg2)

            # create relation between current args

            # first, convert relation to all caps and join by underscore, removing apostrophes from contractions
            converted_relation = '_'.join([word for word in relation.upper().split()])
            # then, grab current args and create a relation. if exists, does nothing due to MERGE
            sess.run("MATCH (a0:arg1), (a1:arg2) "
                     "WHERE a0.text = '%s' AND a1.text = '%s' "
                     "MERGE (a0)-[:%s]->(a1)" % (arg1, arg2, converted_relation))
