# python 3
# this script should be in the same directory as the folder entitled ClausIE

import os
import re
import matplotlib.pyplot as plt
import operator
import sys


def read_extraction_file(output):
    extraction_idx = {}

    with open(output, encoding='latin-1') as f:
        for line in f:
            # split line by tabs
            ln = line.strip().split('\t')
            # In the case the line in the file is the original sentence, skip the line.
            # Otherwise, create an index for the sentences to be used later.
            if len(ln) == 1:
                continue
            else:
                # break apart the line into sentence id, extraction and confidence
                sent_id = int(ln[0])
                extraction = tuple(ln[1:-1])
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
    corr, incorr, unknown = {}, {}, {}

    for e_id, ex in extractions.items():
        corr[e_id], incorr[e_id], unknown[e_id] = [], [], []
        if e_id not in gold:
            gold[e_id] = {}
        for e in ex:
            confidence = extractions[e_id][e]
            if len(e) == 3:
                arg1, rel, arg2 = e
                out_line = str(e_id) + '\t' + str(arg1) + '\t' + str(rel) + '\t' + str(arg2) + '\n'
            elif len(e) == 2:
                arg1, rel = e
                out_line = str(e_id) + '\t' + str(arg1) + '\t' + str(rel) + '\t' + "" + '\n'
            else:
                continue

            if e in gold[e_id]:
                # if the extraction is in gold standard and marked as correct
                if gold[e_id][e] == 1:
                    corr_and_conf.append((1, confidence))
                    corr[e_id].append(out_line)
                # if the extraction is in gold standard and marked as incorrect
                else:
                    corr_and_conf.append((0, confidence))
                    incorr[e_id].append(out_line)
            # if the extraction is not found in the gold standard
            else:
                unknown[e_id].append(out_line)

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

    print('precision =', num_correct, '/', num_extractions, '=', num_correct / num_extractions)
    return prec_by_extr, corr, incorr, unknown


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
            sentence_index = {id:sent for id, sent in enumerate([line for line in f
                                                                 if len(line.strip().split('\t')) == 1])}
        data_set_name = re.findall(r'ClausIE/(.*?)/', path)[0]
        for system, out_file in sorted(data.items(), key=operator.itemgetter(0)):
            # skip over the all files
            if 'all' not in out_file:
                # fetch gold standard path and read its data into a dictionary
                gold_index = read_extraction_file(path_to_gold)
                # read the output file and collect all the data into a dictionary
                extraction_index = read_extraction_file(path + out_file)
                print('for the system ' + system + ', on the ' + data_set_name + ' data set:')
                # compare a system's output against the gold standard
                precision_by_extraction, corrects, incorrects, unknowns = compare(gold_index, extraction_index)
                # plot results, making the Nemex lines stand out more, and write Nemex results to file
                if 'nemex' in system:
                    line_width = 2.0
                    line_style = 'solid'
                    write_eval_results(corrects, incorrects, unknowns,
                                       output_folder, data_set_name, system, sentence_index)
                else:
                    line_width = 1.0
                    line_style = 'dashed'
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

        plt.savefig(output_folder + data_set_name + '/plot.png')
        plt.clf()
        print('\n*********************************************\n')
