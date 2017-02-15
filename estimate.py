from collections import Counter
import math
import operator


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

    tfs = {sent: {word: count / sum(counter.values()) for word, count in counter.items()}
           for sent, counter in counts_by_sent.items()}

    idfs = {term: math.log(num_docs / freq) for term, freq in num_docs_with_term.items()}

    return tfs, idfs


def cos_sim(corr_tfs, corr_idfs, incorr_tfs, incorr_idfs, unannotated_unknowns):
    anno_known = {i: {e: 0 for e in extracts} for i, extracts in unannotated_unknowns.items()}
    no_corr, no_incorr, no_either = 0, 0, 0
    new_corr = {}
    new_incorr = {}

    for i, extracts in sorted(unannotated_unknowns.items()):
        new_corr[i] = {}
        new_incorr[i] = {}
        for id, extract in enumerate(extracts):
            query = []
            corr_buff = []
            incorr_buff = []
            for arg in extract:
                words = [word.lower().strip('"') for word in arg.split()]
                query += [word for word in words if word]
            query_tfs = {term: count / len(query) for term, count in Counter(query).items()}
            corr_query_tfidfs, incorr_query_tfidfs = [], []

            for term, tf in query_tfs.items():
                if term not in corr_idfs:
                    corr_query_tfidfs.append((term, 0))
                else:
                    corr_query_tfidfs.append((term, tf * corr_idfs[term]))
                if term not in incorr_idfs:
                    incorr_query_tfidfs.append((term, 0))
                else:
                    incorr_query_tfidfs.append((term, tf * incorr_idfs[term]))

            corr_doc_tfidfs = {term: tf * corr_idfs[term] for term, tf in corr_tfs[i].items()}
            incorr_doc_tfidfs = {term: tf * incorr_idfs[term] for term, tf in incorr_tfs[i].items()}

            for term, val in corr_query_tfidfs:
                if term not in corr_doc_tfidfs:
                    corr_doc_tfidfs[term] = 0
                    corr_buff.append(term)
            for term, val in incorr_query_tfidfs:
                if term not in incorr_doc_tfidfs:
                    incorr_doc_tfidfs[term] = 0
                    incorr_buff.append(term)

            corr_ab_sum = sum([val * corr_doc_tfidfs[term] for term, val in corr_query_tfidfs])
            corr_mag_a = math.sqrt(sum([v**2 for v in corr_doc_tfidfs.values()]))
            corr_mag_b = math.sqrt(sum([t[1]**2 for t in corr_query_tfidfs]))

            incorr_ab_sum = sum([val * incorr_doc_tfidfs[term] for term, val in incorr_query_tfidfs])
            incorr_mag_a = math.sqrt(sum([v**2 for v in incorr_doc_tfidfs.values()]))
            incorr_mag_b = math.sqrt(sum([t[1]**2 for t in incorr_query_tfidfs]))

            if corr_mag_a and corr_mag_b and incorr_mag_a and incorr_mag_b:
                corr_sim = corr_ab_sum / (corr_mag_a * corr_mag_b)
                incorr_sim = incorr_ab_sum / (incorr_mag_a * incorr_mag_b)
                if corr_sim > incorr_sim:
                    anno_known[i][extract] = 1
                    new_corr[i][id] = corr_buff
                else:
                    anno_known[i][extract] = 0
                    new_incorr[i][id] = incorr_buff
            elif corr_mag_a and corr_mag_b and not incorr_mag_a or not incorr_mag_b:
                no_incorr += 1
                anno_known[i][extract] = 1
                new_corr[i][id] = corr_buff
            elif incorr_mag_a and incorr_mag_b and not corr_mag_a or not corr_mag_b:
                no_corr += 1
                anno_known[i][extract] = 0
                new_incorr[i][id] = incorr_buff
            else:
                no_either += 1
                anno_known[i][extract] = 0.5

    return anno_known, no_corr, no_incorr, no_either, new_corr, new_incorr
