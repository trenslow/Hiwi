from collections import Counter
import math


def estimate_weights(corrects, incorrects, unknowns):
    corr_tfs, corr_idfs = calculate_tfs_idfs(corrects)
    incorr_tfs, incorr_idfs = calculate_tfs_idfs(incorrects)
    return cos_sim(corr_tfs, corr_idfs, incorr_tfs, incorr_idfs, unknowns)


def calculate_tfs_idfs(outputs):
    # treating sentences as a document
    counts_by_sent = {}
    num_docs_with_term = Counter()
    num_docs = len(outputs)

    for i, extrs in outputs.items():
        counts_by_sent[i] = Counter()
        buffer = set()
        # split up extractions
        for ex in extrs:
            # split up args
            for e in ex:
                # split up words
                e = e.replace('[attrib=', '')
                e = e.replace('[enabler=', '')
                e = e.replace(']', '')
                up = e.strip('"')
                words = [u.lower() for u in up.split()]
                counts_by_sent[i].update(words)
                buffer.update(words)
        num_docs_with_term.update(buffer)

    tfs = {i: {word: count / sum(counter.values()) for word, count in counter.items()}
           for i, counter in counts_by_sent.items()}

    idfs = {term: 1 + math.log(num_docs / freq) for term, freq in num_docs_with_term.items()}

    return tfs, idfs


def cos_sim(corr_tfs, corr_idfs, incorr_tfs, incorr_idfs, unannotated_unknowns):
    anno_known = {i: {e: 0 for e in extracts} for i, extracts in unannotated_unknowns.items()}

    for i, extracts in unannotated_unknowns.items():
        for extract in extracts:
            query = []
            for arg in extract:
                words = [word.lower().strip('"') for word in arg.split()]
                query += words
            query_tfs = {term: count / len(query) for term, count in Counter(query).items()}
            corr_query_tfidfs, incorr_query_tfidfs = {}, {}

            for term, tf in query_tfs.items():
                if term not in corr_idfs:
                    corr_query_tfidfs[term] = 0
                else:
                    corr_query_tfidfs[term] = tf * corr_idfs[term]
                if term not in incorr_idfs:
                    incorr_query_tfidfs[term] = 0
                else:
                    incorr_query_tfidfs[term] = tf * incorr_idfs[term]

            corr_doc_tfidfs = {term: tf * corr_idfs[term] for term, tf in corr_tfs[i].items()}
            incorr_doc_tfidfs = {term: tf * incorr_idfs[term] for term, tf in incorr_tfs[i].items()}

            for term, val in corr_query_tfidfs.items():
                if term not in corr_doc_tfidfs:
                    corr_doc_tfidfs[term] = 0
            for term, val in incorr_query_tfidfs.items():
                if term not in incorr_doc_tfidfs:
                    incorr_doc_tfidfs[term] = 0

            corr_ab_sum = sum([val * corr_doc_tfidfs[term] for term, val in corr_query_tfidfs.items()])
            corr_mag_a = math.sqrt(sum([v**2 for v in corr_doc_tfidfs.values()]))
            corr_mag_b = math.sqrt(sum([v**2 for v in corr_query_tfidfs.values()]))

            incorr_ab_sum = sum([val * incorr_doc_tfidfs[term] for term, val in incorr_query_tfidfs.items()])
            incorr_mag_a = math.sqrt(sum([v**2 for v in incorr_doc_tfidfs.values()]))
            incorr_mag_b = math.sqrt(sum([v**2 for v in incorr_query_tfidfs.values()]))

            if corr_mag_a and corr_mag_b and incorr_mag_a and incorr_mag_b:
                corr_sim = corr_ab_sum / (corr_mag_a * corr_mag_b)
                incorr_sim = incorr_ab_sum / (incorr_mag_a * incorr_mag_b)
                if corr_sim > incorr_sim:
                    anno_known[i][extract] = 1
                else:
                    anno_known[i][extract] = 0
            else:
                anno_known[i][extract] = 0.5

    return anno_known
