# python 3
# this script should be in the same directory as the folder entitled ClausIE

import os
import re
import matplotlib.pyplot as plt
import operator
import sys


def read_gold_standard(gold):
    gold_idx = dict()

    with open(gold, encoding='latin-1') as f:
        for line in f:
            # split line by tabs
            ln = line.strip().split('\t')
            # in the case the line in the file is the original sentence, skip the line
            if len(ln) == 1:
                continue
            else:
                # break apart the line into sentence id, extraction and rating
                sent_id = int(ln[0])
                extraction = tuple(ln[1:-1])
                rating = int(ln[-1])
                # initialize the gold index entry with the index as key and empty dict as value
                if sent_id not in gold_idx:
                    gold_idx[sent_id] = dict()
                # initialize the dict for each gold index with extraction as key and rating as value
                if extraction not in gold_idx[sent_id]:
                    gold_idx[sent_id][extraction] = dict()
                # add each relation to the gold index
                gold_idx[sent_id][extraction] = rating

    return gold_idx


def read_extraction_file(output):
    extract_idx = dict()

    with open(output, encoding='latin-1') as f:
        for line in f:
            # split line by tabs
            ln = line.strip().split('\t')
            # in the case the line in the file is the original sentence, skip the line
            if len(ln) == 1:
                continue
            else:
                # break apart the line into sentence id, extraction and confidence
                sent_id = int(ln[0])
                extraction = tuple(ln[1:-1])
                conf = float(ln[-1])
                # initialize the extraction index entry with the index as key and empty dict as value
                if sent_id not in extract_idx:
                    extract_idx[sent_id] = dict()
                # initialize the dict for each extraction index with extraction as key and confidence as value
                if extraction not in extract_idx[sent_id]:
                    extract_idx[sent_id][extraction] = dict()
                # add each relation to the extraction index with its corresponding confidence value
                extract_idx[sent_id][extraction] = conf

    return extract_idx


def compare(gold, output):
    corr_and_conf = []
    not_in_gold = []

    for o_id, o_ex in output.items():
        for ex in o_ex:
            confidence = output[o_id][ex]
            if o_id not in gold:
                gold[o_id] = dict()

            if ex in gold[o_id]:
                if gold[o_id][ex] == 1:
                    corr_and_conf.append((1, confidence))
                else:
                    corr_and_conf.append((0, confidence))
            else:
                not_in_gold.append((o_id, ex))

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
    return prec_by_extr


def graph(dat, color, style, sys_name, width, data_name, xlim):
    plt.plot(dat, color=color, linestyle=style, label=sys_name, linewidth=width)
    plt.xlabel('Number of extractions')
    plt.ylabel('Precision')
    plt.title(data_name + ' data set')
    plt.xlim(-.05 * xlim, xlim)
    plt.ylim(-0.05, 1.05)
    plt.legend(loc='lower right', framealpha=0.5)
    if sys.platform.startswith('linux'):
        mng = plt.get_current_fig_manager()
        mng.window.showMaximized()


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
    if sys.platform.startswith('linux'):
        mng = plt.get_current_fig_manager()
        mng.window.showMaximized()


if __name__ == '__main__':
    system_name = r'extractions-(.*?).txt'
    nyt_folder = 'ClauseIE/nyt/'
    reverb_folder = 'ClauseIE/reverb/'
    wiki_folder = 'ClauseIE/wikipedia/'
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
        for system, out_file in sorted(data.items(), key=operator.itemgetter(0)):
            # skip over the all files
            if 'all' not in out_file:
                # fetch gold standard path and read its data into a dictionary
                gold_file = data['all-labeled']
                gold_index = read_gold_standard(path + gold_file)
                # read the output file and collect all the data into a dictionary
                extraction_index = read_extraction_file(path + out_file)
                data_set_name = re.findall(r'ClauseIE/(.*?)/', path)[0]
                print('for the system ' + system + ', on the ' + data_set_name + ' data set:')
                # compare a system's output against the gold standard
                precision_by_extraction = compare(gold_index, extraction_index)
                total_extractions = len(precision_by_extraction)

                # plot results, making the Nemex lines stand out more
                if 'nemex' in system:
                    line_width = 2.0
                    line_style = 'solid'
                else:
                    line_width = 1.0
                    line_style = 'dashed'
                # for plots analogous to those on the ClauseIE paper
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

        plt.show()
        plt.clf()
        print('\n*********************************************\n')
