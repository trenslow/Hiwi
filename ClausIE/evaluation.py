#!/usr/bin/python3
# this script should be in the same directory as the folder entitled ClausIE
import os
import re
import matplotlib.pyplot as plt
import operator
from estimate import estimate_weights
from collections import OrderedDict as OD


def read_extraction_file(file):
    extraction_idx = OD()
    sent_id = -1

    with open(file, encoding='latin-1') as f:
        for line in f:
            # split line by tabs
            ln = line.strip().split('\t')
            # In the case the line in the file is the original sentence, skip the line.
            # Otherwise, create an index for the sentences to be used later.
            if len(ln) == 1:
                sent_id += 1
                extraction_idx[sent_id] = OD()
                continue
            else:
                # break apart the line into sentence id, extraction and confidence
                sent_id = int(ln[0])
                extraction = tuple([l.replace(' , ', ' ') for l in ln[1:-1]])
                conf = float(ln[-1])
                # initialize the dict for each extraction index with extraction as key and confidence as value
                if extraction not in extraction_idx[sent_id]:
                    extraction_idx[sent_id][extraction] = OD()
                # add each relation to the extraction index with its corresponding confidence value
                extraction_idx[sent_id][extraction] = conf

    return extraction_idx


def compare(gold, extractions, nemex):
    corr_and_conf = []
    corr, incorr, unknwn = OD(), OD(), OD()

    for sent_id, extr in extractions.items():
        corr[sent_id] = OD()
        incorr[sent_id] = OD()
        unknwn[sent_id] = []
        for ext in extr:
            weight = extractions[sent_id][ext]
            # if the extraction is in gold standard and marked as correct
            if ext in gold[sent_id] and gold[sent_id][ext] == 1:
                corr_and_conf.append((1, weight))
                corr[sent_id][ext] = weight
            # if the extraction is in gold standard and marked as incorrect
            elif ext in gold[sent_id] and gold[sent_id][ext] == 0:
                corr_and_conf.append((0, weight))
                incorr[sent_id][ext] = weight
            # if the extraction is not found in the gold standard
            else:
                unknwn[sent_id].append(ext)

    # initialize the count of correct extractions and total extractions
    # initialize a list to collect precision values for the k-th extraction to be graphed
    num_correct, num_extractions = 0, 0
    prec_by_extr = []
    # sort the correct and confidence tuple list by descending confidence values
    sort_c_and_c = sorted(corr_and_conf, key=operator.itemgetter(1), reverse=True)
    for k in range(len(sort_c_and_c)):
        # increase the count of correct extractions and total number of extractions
        num_correct += sort_c_and_c[k][0]
        num_extractions = k + 1
        # calculate precision values for the top k-th extraction and collect in list
        precision = num_correct / num_extractions
        prec_by_extr.append(precision)

    if nemex:
        return prec_by_extr, corr, incorr, unknwn
    else:
        return prec_by_extr, corr, incorr


def write_nemex_results(corrs, incorrs, unkwns, out_folder, dat_set, system, sent_idx):
    path_to_txt = out_folder + dat_set + '/'
    correct_file = open(path_to_txt + system + '_' + 'correct.txt', 'w+')
    incorrect_file = open(path_to_txt + system + '_' + 'incorrect.txt', 'w+')
    unknown_file = open(path_to_txt + system + '_' + 'unknown.txt', 'w+')

    for id, sent in sorted(sent_idx.items(), key=operator.itemgetter(0)):
        correct_file.write(sent)
        incorrect_file.write(sent)
        unknown_file.write(sent)

        if corrs.get(id):
            for ex in corrs[id]:
                correct_file.write(str(id) + '\t' + '\t'.join(ex) + '\t' + str(corrs[id][ex]) + '\n')

        if incorrs.get(id):
            for ex in incorrs[id]:
                incorrect_file.write(str(id) + '\t' + '\t'.join(ex) + '\t' + str(incorrs[id][ex]) + '\n')

        if unkwns.get(id):
            for ex in unkwns[id]:
                unknown_file.write(str(id) + '\t' + '\t'.join(ex) + '\t' + str(unkwns[id][ex]) + '\n')

    correct_file.close()
    incorrect_file.close()
    unknown_file.close()


def write_stats_file(out_folder, data_set, system, num_gold_corrs, num_gold_incorrs,
                     num_nemex_corrs, num_nemex_incorrs, num_nemex_unknwns):
    path_to_file = out_folder + data_set + '/'
    with open(path_to_file + system + '_' + 'stats.txt', 'w+') as stats_file:
        num_extractions = num_nemex_corrs + num_nemex_incorrs
        stats_file.write('For the system ' + system + ', on the ' + data_set + ' data set:' + '\n')
        stats_file.write('# of nemex positives / # of total positives = ' + str(num_nemex_corrs) + ' / ' +
                         str(num_gold_corrs) + ' = ' + '{0:.2%}'.format(num_nemex_corrs / num_gold_corrs) + '\n')
        stats_file.write('# of nemex negatives / # of total negatives = ' + str(num_nemex_incorrs) + ' / ' +
                         str(num_gold_incorrs) + ' = ' + '{0:.2%}'.format(num_nemex_incorrs / num_gold_incorrs) + '\n')
        stats_file.write('# of nemex unknowns: ' + str(num_nemex_unknwns) + '\n')
        stats_file.write('precision = ' + str(num_nemex_corrs) + ' / ' + str(num_extractions) +
                         ' = ' + '{0:.2%}'.format(num_nemex_corrs / num_extractions))


def graph(dat, color, style, sys_name, width, data_name, xlim):
    plt.plot(dat, color=color, linestyle=style, label=sys_name, linewidth=width)
    plt.xlabel('Number of extractions')
    plt.ylabel('Precision')
    plt.title(data_name + ' data set')
    plt.xlim(-.05 * xlim, xlim)
    plt.ylim(-0.05, 1.05)
    plt.legend(loc='lower right', framealpha=0.5)


def graph_subplots(plot_num, dat, color, style, sys_name, width, test, data_name, xlim):
    plt.subplot(2, 1, plot_num)
    plt.subplots_adjust(hspace=0.35)
    plt.plot(dat, color=color, linestyle=style, label=sys_name, linewidth=width)
    plt.xlabel('Number of extractions')
    plt.ylabel('Precision')
    if test:
        plt.title(data_name + ' test data set')
    else:
        plt.title(data_name + ' development data set')
    plt.xlim(-.05 * xlim, xlim)
    plt.ylim(-0.05, 1.05)
    plt.legend(loc='lower right', framealpha=0.5)


def create_output_directory(out_folder):
    if not os.path.exists(out_folder + 'nyt/'):
        os.makedirs(out_folder + 'nyt/')
    if not os.path.exists(out_folder + 'reverb/'):
        os.makedirs(out_folder + 'reverb/')
    if not os.path.exists(out_folder + 'wikipedia/'):
        os.makedirs(out_folder + 'wikipedia/')


def write_new_vocab(new, corr, out_fold, dat_set, syst):
    path_to_file = out_fold + dat_set + '/'
    if corr:
        typ = 'correct'
    else:
        typ = 'incorrect'
    with open(path_to_file + syst + '_' + 'new_' + typ + '.txt', 'w+') as n:
        for sent_id, exs in new.items():
            for ex, words in exs.items():
                if words:
                    n.write(str(sent_id) + '-' + str(ex) + '\t' + ','.join(word for word in words) + '\n')


def clean(new_c, unknws):
    all_clean = OD()

    for idx, ext in new_c.items():
        all_clean[idx] = []

        for e, tokens in ext.items():
            if tokens:
                uncleaned_ext = [arg.strip('"').split(' ') for arg in unknws[idx][e]]
                buffer = []
                for arg in uncleaned_ext:
                    buffer.append([a for a in arg if a not in tokens])

                cleaned_ext = tuple('"' + ' '.join(arg) + '"' for arg in buffer if arg)
                if len(cleaned_ext) == 3:
                    all_clean[idx].append(cleaned_ext)

    return all_clean


if __name__ == '__main__':
    system_regex = r'extractions-(.*?).txt'
    nyt_folder = 'ClausIE/nyt/'
    reverb_folder = 'ClausIE/reverb/'
    wiki_folder = 'ClausIE/wikipedia/'
    output_folder = 'nemexOutputs/'
    create_output_directory(output_folder)
    full_plots = True
    do_learn = True

    # index that keeps track of all the paths to the output files
    # filters out any files that don't have extractions in them
    output_file_index = {nyt_folder: {re.findall(system_regex, n)[0]: n
                                      for n in os.listdir(nyt_folder) if 'extractions' in n and 'all' not in n},
                         reverb_folder: {re.findall(system_regex, r)[0]: r
                                         for r in os.listdir(reverb_folder) if 'extractions' in r and 'all' not in r},
                         wiki_folder: {re.findall(system_regex, w)[0]: w
                                       for w in os.listdir(wiki_folder) if 'extractions' in w and 'all' not in w}
                         }

    # colors taken from graphs in ClausIE paper, nemex colors chosen for best contrast
    colors = {'clausie': 'blue',
              'clausie-ncc': 'cyan',
              'reverb': 'lime',
              'ollie': 'orange',
              'textrunner': 'red',
              'textrunner_reverb': 'magenta',
              'woe_parse': 'gray',
              'nemex-penn-TD': 'darkgreen',
              'nemex-ud-TD': 'black',
              'nemex-penn-BU': 'purple',
              'nemex-ud-BU': 'gold'
              }

    # loop over all paths in all the folders
    for path, data in sorted(output_file_index.items(), key=operator.itemgetter(0)):
        # initialize the number of max extractions to be used for setting x-axis limits later
        x_limit = 0
        path_to_gold = path + 'extractions-all-labeled.txt'
        with open(path_to_gold, encoding='latin-1') as f:
            sentence_index = {i: sent for i, sent in enumerate([line for line in f
                                                                if len(line.strip().split('\t')) == 1])}
        data_set_name = re.findall(r'ClausIE/(.*?)/', path)[0]
        for system, out_file in sorted(data.items(), key=operator.itemgetter(0)):
            # read gold standard data into dictionary
            gold_index = read_extraction_file(path_to_gold)
            gold_corrects = {i: {extr: weight for extr, weight in extrs.items() if weight == 1}
                             for i, extrs in gold_index.items()}
            gold_incorrects = {i: {extr: weight for extr, weight in extrs.items() if weight == 0}
                               for i, extrs in gold_index.items()}
            # read a system's output into a dictionary
            extraction_index = read_extraction_file(path + out_file)
            # plot results, making the Nemex lines stand out more, and write Nemex results to file
            if 'nemex' in system:
                line_width = 2.0
                line_style = 'solid'
                # compare a nemex output against the gold standard
                precision_by_extraction, corrects, incorrects, unknowns = compare(gold_index,
                                                                                  extraction_index, True)
                unknowns_and_weights, no_corr, no_incorr, no_either, new_corr, new_incorr = estimate_weights(
                    gold_corrects, gold_incorrects, unknowns)

                write_new_vocab(new_corr, True, output_folder, data_set_name, system)
                write_new_vocab(new_incorr, False, output_folder, data_set_name, system)

                if do_learn:
                    all_cleaned = clean(new_corr, unknowns)
                    learned_corrects = OD()
                    for i, c_extrs in all_cleaned.items():
                        learned_corrects[i] = []
                        for c_extr in c_extrs:
                            if c_extr in corrects[i].keys():
                                learned_corrects[i].append(c_extr)
                    print('for the system: ' + system + ' on data set ' + data_set_name)
                    print('# of correct learned extractions:', sum([len(extrs) for extrs in learned_corrects.values()]))


                num_unknowns = sum([len(e) for e in unknowns.values()])
                with open(output_folder + data_set_name + '/' + system + '_estimation.txt', 'w+') as est_results:
                    est_results.write('total unknown extractions: ' + str(num_unknowns) + '\n')
                    est_results.write('no correct doc or query:(labeled as 0) ' + str(no_corr) + ' / ' +
                                      str(num_unknowns) + ' = ' + str(no_corr / num_unknowns) + '\n')
                    est_results.write('no incorrect doc or query:(labeled as 1) ' + str(no_incorr) + ' / ' +
                                      str(num_unknowns) + ' = ' + str(no_incorr / num_unknowns) + '\n')
                    est_results.write('no correct and no incorrect doc or query:(labeled as 0.5) ' + str(no_either) +
                                      ' / ' + str(num_unknowns) + ' = ' + str(no_either / num_unknowns))

                num_gold_corrects = sum([len(lst) for lst in gold_corrects.values()])
                num_gold_incorrects = sum([len(lst) for lst in gold_incorrects.values()])
                num_nemex_corrects = sum([len(lst) for lst in corrects.values()])
                num_nemex_incorrects = sum([len(lst) for lst in incorrects.values()])
                num_nemex_unknowns = sum([len(lst) for lst in unknowns.values()])

                write_nemex_results(corrects, incorrects, unknowns_and_weights, output_folder,
                                   data_set_name, system, sentence_index)
                write_stats_file(output_folder, data_set_name, system,
                                 num_gold_corrects, num_gold_incorrects,
                                 num_nemex_corrects, num_nemex_incorrects, num_nemex_unknowns)

            else:
                line_width = 1.0
                line_style = 'dashed'
                # compare a system's output against the gold standard
                precision_by_extraction, corrects, incorrects = compare(gold_index, extraction_index, False)

            # for plots analogous to those on the ClausIE paper
            total_extractions = len(precision_by_extraction)
            if full_plots:
                if total_extractions > x_limit:
                    x_limit = total_extractions
                graph(precision_by_extraction, colors[system], line_style, system, line_width,
                      data_set_name, x_limit)
            # split the plot into development and test sets
            else:
                # slice the data into development and test sets
                # development set is the first half of data, test set is last half
                slice_idx = total_extractions // 2
                dev_set = precision_by_extraction[:slice_idx]
                test_set = precision_by_extraction[slice_idx:]
                if slice_idx + 1 > x_limit:
                    x_limit = slice_idx + 1
                graph_subplots(1, dev_set, colors[system], line_style, system, line_width,
                               False, data_set_name, x_limit)
                graph_subplots(2, test_set, colors[system], line_style, system, line_width,
                               True, data_set_name, x_limit)

        plt.savefig(output_folder + data_set_name + '/' + data_set_name + '_plot.pdf', format='pdf', dpi=1200)
        plt.show()
        plt.clf()