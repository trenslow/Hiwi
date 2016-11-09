#!/usr/bin/python3
# this script should be in the same directory as the folder entitled ClausIE

import os
import re
import matplotlib.pyplot as plt
import operator
import string


def read_extraction_file(output):
    extraction_idx = {}
    punct = {","}

    with open(output, encoding='latin-1') as file:
        for line in file:
            # split line by tabs
            ln = line.strip().split('\t')
            # In the case the line in the file is the original sentence, skip the line.
            # Otherwise, create an index for the sentences to be used later.
            if len(ln) == 1:
                continue
            else:
                # break apart the line into sentence id, extraction and confidence
                sent_id = int(ln[0])
                extraction = tuple([''.join(el for el in l if el not in punct) for l in ln[1:-1]])
                conf = float(ln[-1])
                # initialize the extraction index entry with the index as key and empty dict as value
                if sent_id not in extraction_idx:
                    extraction_idx[sent_id] = {}
                # initialize the dict for each extraction index with extraction as key and confidence as value
                if extraction not in extraction_idx[sent_id]:
                    extraction_idx[sent_id][extraction] = {}
                # add each relation to the extraction index with its corresponding confidence value
                extraction_idx[sent_id][extraction] = conf

    return extraction_idx


def compare(gold, extractions):
    corr_and_conf = []
    corr, incorr, unknown = {e: [] for e in extractions}, {e: [] for e in extractions}, {e: [] for e in extractions}
    gold_corr, gold_incorr = {g: [] for g in gold}, {g: [] for g in gold}

    for sent_id, extr in extractions.items():
        if sent_id not in gold:
            gold[sent_id] = {}

        for ext in extr:
            weight = extractions[sent_id][ext]
            out_line = str(sent_id) + '\t' + '\t'.join(ext) + '\t' + str(weight) + '\n'
            if ext in gold[sent_id]:
                # if the extraction is in gold standard and marked as correct
                if gold[sent_id][ext] == 1:
                    corr_and_conf.append((1, weight))
                    corr[sent_id].append(out_line)
                # if the extraction is in gold standard and marked as incorrect
                else:
                    corr_and_conf.append((0, weight))
                    incorr[sent_id].append(out_line)
            # if the extraction is not found in the gold standard
            else:
                unknown[sent_id].append(out_line)

        extras = set(gold[sent_id].keys()).symmetric_difference(set(extr.keys()))
        gold_corr[sent_id] = [str(sent_id) + '\t' + '\t'.join(e) + '\n'
                              for e in extras if e in gold[sent_id] and gold[sent_id][e] == 1]
        gold_incorr[sent_id] = [str(sent_id) + '\t' + '\t'.join(e) + '\n'
                                for e in extras if e in gold[sent_id] and gold[sent_id][e] == 0]

    return corr_and_conf, corr, incorr, unknown, gold_corr, gold_incorr


def calculate_precision(c_and_c):
    # initialize the count of correct extractions and total extractions
    # initialize a list to collect precision values for the k-th extraction to be graphed
    num_correct, num_extractions = 0, 0
    prec_by_extr = []
    # sort the correct and confidence tuple list by descending confidence values
    sort_c_and_c = sorted(c_and_c, key=operator.itemgetter(1), reverse=True)
    for k in range(len(sort_c_and_c)):
        # increase the count of correct extractions and total number of extractions
        num_correct += sort_c_and_c[k][0]
        num_extractions = k + 1
        # calculate precision values for the top k-th extraction and collect in list
        precision = num_correct / num_extractions
        prec_by_extr.append(precision)

    return prec_by_extr


def write_eval_results(corrs, incorrs, unkwns, out_folder, dat_set, system, sent_idx):
    path_to_txt = out_folder + dat_set + '/'
    correct_file = open(path_to_txt + system + '_' + 'correct.txt', 'w+')
    incorrect_file = open(path_to_txt + system + '_' + 'incorrect.txt', 'w+')
    unknown_file = open(path_to_txt + system + '_' + 'unknown.txt', 'w+')

    for id, sent in sorted(sent_idx.items(), key=operator.itemgetter(0)):
        correct_file.write(sent)
        incorrect_file.write(sent)
        unknown_file.write(sent)
        if id in corrs:
            correct_file.writelines(corrs[id])
        if id in incorrs:
            incorrect_file.writelines(incorrs[id])
        if id in unkwns:
            unknown_file.writelines(unkwns[id])

    correct_file.close()
    incorrect_file.close()
    unknown_file.close()


def write_gold_extras(gold_corrs, gold_incorrs, out_folder, dat_set, system, sent_idx):
    path_to_file = out_folder + dat_set + '/'
    gold_corr_file = open(path_to_file + system + '_' + 'goldCorrect.txt', 'w+')
    gold_incorr_file = open(path_to_file + system + '_' + 'goldIncorrect.txt', 'w+')

    for id, sent in sorted(sent_idx.items(), key=operator.itemgetter(0)):
        gold_corr_file.write(sent)
        gold_incorr_file.write(sent)
        if id in gold_corrs:
            gold_corr_file.writelines(gold_corrs[id])
        if id in gold_incorrs:
            gold_incorr_file.writelines(gold_incorrs[id])

    gold_corr_file.close()
    gold_incorr_file.close()


def write_stats_file(out_folder, dat_set, system, num_gold_corrs, num_gold_incorrs,
                     num_nemex_corrs, num_nemex_incorrs, num_nemex_unknwns):
    path_to_file = out_folder + dat_set + '/'
    stats_file = open(path_to_file + system + '_' + 'stats.txt', 'w+')
    num_extractions = num_nemex_corrs + num_nemex_incorrs

    stats_file.write('For the system ' + system + ', on the ' + dat_set + ' data set:' + '\n')
    stats_file.write('# of nemex positives / # of total positives = ' + str(num_nemex_corrs) + ' / ' +
                     str(num_gold_corrs) + ' = ' + '{0:.2%}'.format(num_nemex_corrs / num_gold_corrs) + '\n')
    stats_file.write('# of nemex negatives / # of total negatives = ' + str(num_nemex_incorrs) + ' / ' +
                     str(num_gold_incorrs) + ' = ' + '{0:.2%}'.format(num_nemex_incorrs / num_gold_incorrs) + '\n')
    stats_file.write('# of nemex unknowns: ' + str(num_nemex_unknwns) + '\n')
    stats_file.write('precision = ' + str(num_nemex_corrs) + ' / ' + str(num_extractions) +
                     ' = ' + '{0:.2%}'.format(num_nemex_corrs / num_extractions))

    stats_file.close()


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


if __name__ == '__main__':
    system_name = r'extractions-(.*?).txt'
    nyt_folder = 'ClausIE/nyt/'
    reverb_folder = 'ClausIE/reverb/'
    wiki_folder = 'ClausIE/wikipedia/'
    output_folder = 'nemexOutputs/'
    create_output_directory(output_folder)
    full_plots = True

    # index that keeps track of all the paths to the output files
    # filters out any files that don't have extractions in them
    output_file_index = {nyt_folder: {re.findall(system_name, n)[0]: n
                                      for n in os.listdir(nyt_folder) if 'extractions' in n},
                         reverb_folder: {re.findall(system_name, r)[0]: r
                                         for r in os.listdir(reverb_folder) if 'extractions' in r},
                         wiki_folder: {re.findall(system_name, w)[0]: w
                                       for w in os.listdir(wiki_folder) if 'extractions' in w}
                         }

    # colors taken from graphs in ClausIE paper
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
        path_to_gold = path + data['all-labeled']
        with open(path_to_gold, encoding='latin-1') as f:
            sentence_index = {i: sent for i, sent in enumerate([line for line in f
                                                                if len(line.strip().split('\t')) == 1])}
        data_set_name = re.findall(r'ClausIE/(.*?)/', path)[0]
        for system, out_file in sorted(data.items(), key=operator.itemgetter(0)):
            # skip over the all files
            if 'all' not in out_file:
                # fetch gold standard path and read its data into a dictionary
                gold_index = read_extraction_file(path_to_gold)
                # read the output file and collect all the data into a dictionary
                extraction_index = read_extraction_file(path + out_file)
                # plot results, making the Nemex lines stand out more, and write Nemex results to file
                if 'nemex' in system:
                    line_width = 2.0
                    line_style = 'solid'
                    # compare a nemex's output against the gold standard
                    correct_and_confidence, corrects, incorrects,\
                    unknowns, gold_corrects, gold_incorrects = compare(gold_index, extraction_index)
                    num_gold_corrects = sum([len(lst) for lst in gold_corrects.values()])
                    num_gold_incorrects = sum([len(lst) for lst in gold_incorrects.values()])
                    num_nemex_corrects = sum([len(lst) for lst in corrects.values()])
                    num_nemex_incorrects = sum([len(lst) for lst in incorrects.values()])
                    num_nemex_unknowns = sum([len(lst) for lst in unknowns.values()])

                    write_eval_results(corrects, incorrects, unknowns, output_folder,
                                       data_set_name, system, sentence_index)
                    write_gold_extras(gold_corrects, gold_incorrects, output_folder,
                                      data_set_name, system, sentence_index)
                    write_stats_file(output_folder, data_set_name, system,
                                     num_gold_corrects, num_gold_incorrects,
                                     num_nemex_corrects, num_nemex_incorrects, num_nemex_unknowns)

                else:
                    line_width = 1.0
                    line_style = 'dashed'
                    # compare a system's output against the gold standard
                    correct_and_confidence, corrects, incorrects,\
                    unknowns, gold_corrects, gold_incorrects = compare(gold_index, extraction_index)

                precision_by_extraction = calculate_precision(correct_and_confidence)
                # for plots analogous to those on the ClausIE paper
                total_extractions = len(precision_by_extraction)
                if full_plots:
                    if total_extractions > x_limit:
                        x_limit = total_extractions
                    graph(precision_by_extraction, colors[system], line_style, system, line_width,
                          data_set_name, x_limit)
                # plot a development and test set
                else:
                    # slice the data into development and test sets
                    # development set is the first half of data, test half is last half
                    slice_idx = total_extractions // 2
                    dev_set = precision_by_extraction[:slice_idx]
                    test_set = precision_by_extraction[slice_idx:]
                    if slice_idx + 1 > x_limit:
                        x_limit = slice_idx + 1
                    graph_subplots(1, dev_set, colors[system], line_style, system, line_width,
                                   False, data_set_name, x_limit)
                    graph_subplots(2, test_set, colors[system], line_style, system, line_width,
                                   True, data_set_name, x_limit)

        plt.savefig(output_folder + data_set_name + '/plot.pdf', format='pdf', dpi=1200)
        plt.show()
        plt.clf()