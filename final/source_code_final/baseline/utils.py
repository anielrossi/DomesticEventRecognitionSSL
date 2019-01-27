from __future__ import print_function
import json
import collections
import re
import os
import numpy as np
import random
from six import StringIO
from sklearn import tree



def get_all_tags_from_class(class_name, dataset):
	# NOTE: may include duplicates
	tags = list()
	for sound in dataset[class_name]:
		tags += sound['tags']
	return tags

def get_feature_vector_from_tags(tags, prototype_feature_vector):
        feature_vector = np.zeros(len(prototype_feature_vector))
        for tag in tags:
            try:
                pos = int(prototype_feature_vector.index(tag))
                feature_vector[pos] = 1
            except:
                pass
        return feature_vector

def get_tags_from_feature_vector(feature_vector, prototype_feature_vector):
    tags = []
    for count, element in enumerate(feature_vector):
        if element == 1:
            tag = prototype_feature_vector[int(count)]
            tags.append(tag)
    return tags

def print_most_common_tags(tags, N=15, html=False):
    c = collections.Counter(tags)
    html = "<table><tr><td><b>Tag</b></td><td><b>Count</b></td></tr>"
    for tag, count in c.most_common(N):
        if not html:
            print('%15s  %i' % (tag, count))
        else:
            html += "<tr><td>%s</td><td>%i</td></tr>" % (tag, count)
    html += "</table>"
    return html

def compare_lists(list_a, list_b):
	# Returns True if both lists have the same elements (potentially sorted differently). 
	# Returns False otherwise.
	return len(set(list_a).intersection(list_b)) == len(list_a)

def save_to_json(path="", data=""):
    with open(path, mode='w') as f:
        json.dump(data,f,indent=4)

def load_from_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def print_confussion_matrix(mtx,labels, L=10):
    print("".ljust(L) + "".ljust(((len(mtx[0]) * L) / 2) - 5) + "Predicted")
    print("".ljust(L) + '-' * (len(mtx[0]) * L))
    line = ""
    for label in [""] + labels + ["N instances","P","R","F"]:
        line += label.ljust(L)
    print(line)

    for i in range(0,len(mtx)):
        line = labels[i].ljust(L)
        good = 0
        bad = 0
        bad2 = 0
        for j in range(0,len(mtx[i])):
            line += ("%i" % mtx[i][j]).ljust(L)
            if i == j:
                good += mtx[i][j]
            else:
                bad += mtx[i][j]
                bad2 += mtx[j][i]

        line += ("%i" % (good+bad)).ljust(L)
        
        if good == bad == 0 or good == bad2 == 0:
            precision = -1
            recall = -1
        else:
            precision = float(good)/(good+bad2)
            recall = float(good)/(good+bad)

        # Precision
        line += ("%.4f" % precision).ljust(L)
        # Recall
        line += ("%.4f" % recall).ljust(L)
        # F
        f = 2*((precision * recall)/(precision + recall))
        line += ("%.4f" % f).ljust(L)
        print(line)
    print()

def get_sound_embed_html(freesound_id):
    return '<iframe frameborder="0" scrolling="no" src="http://www.freesound.org/embed/sound/iframe/%i/simple/medium/" width="481" height="86"></iframe>' % freesound_id

def generate_html_with_sound_examples(freesound_ids):
    html = ''
    for sound_id in freesound_ids:
        html += get_sound_embed_html(sound_id)
    return html

def generate_html_tagcloud(tags, N=100, max_px=50, min_px=7, pow_scale=1.8):
    most_common = collections.Counter(tags).most_common(N)
    max_count = max([count for tag, count in most_common])    
    sizes = list()
    for tag, count in most_common:
        size = (10+(max_px*float(count)/max_count))**pow_scale
        sizes.append(size)
    max_size = max(sizes)
    sizes = [int(max_px*float(item)/max_size) for item in sizes]
    sizes = [item if item >= min_px else min_px for item in sizes]
    html_elements = list()
    for (tag, count), size in zip(most_common, sizes):
        html_elements.append('<span style="font-size:%ipx;margin-right:10px;">%s</span> ' % (size, tag))
    random.shuffle(html_elements)
    return ''.join(html_elements)

# For decision trees
def export_tree_as_graph(classifier, feature_names, class_names=True, filename='tree.png'):
    out = StringIO()
    tree.export_graphviz(classifier, 
        out_file=out, 
        feature_names=feature_names, 
        class_names=class_names,
        impurity=True,
        proportion=True)
    fid = open('tmp.dot','w')
    fid.write(out.getvalue())
    fid.close()

    # Print tree with "dot -Tpng tree.dot -o tree.png"
    try:
        os.system('dot -Tpng tmp.dot -o %s' % filename)
    except Exception as e:
        print('ERROR: could not generate %s (%s)' % (filename, e))
    os.system('rm tmp.dot')


def print_tree_as_text(classifier, feature_vector_dimension_labels, class_names):
    tlevel = _tree_rprint('', classifier, feature_vector_dimension_labels, class_names)
    print('<', end='')
    for i in range(3 * tlevel - 2):
        print('-', end='')
    print('>')
    print('Tree Depth: ', tlevel)


"""
treeviz.py

A simple tree visualizer for sklearn DecisionTreeClassifiers.

Written by Lutz Hamel, (c) 2017 - Univeristy of Rhode Island
"""
import operator

def tree_print(clf, X):
    """
    Print the tree of a sklearn DecisionTreeClassifier

    Parameters
    ----------
    clf : DecisionTreeClassifier - A tree that has already been fit.
    X : The original training set
    """
    tlevel = _tree_rprint('', clf, X.columns, clf.classes_)
    print('<',end='')
    for i in range(3*tlevel - 2):
        print('-',end='')
    print('>')
    print('Tree Depth: ',tlevel)

def _tree_rprint(kword, clf, features, labels, node_index=0, tlevel_index=0):
    # Note: The DecisionTreeClassifier uses the Tree structure defined in:
    #       github.com/scikit-learn/scikit-learn/blob/master/sklearn/tree/_tree.pyx
    #       it is an array based tree implementation:

    # indent the nodes according to their tree level
    for i in range(tlevel_index):
        print('  |',end='')

    #  TODO: the following should use the TREE_LEAF constant defined in _tree.pyx
    #        instead of -1, not quite sure how to get at it from the tree user level
    if clf.tree_.children_left[node_index] == -1:  # indicates leaf
        print(kword, end=' ' if kword else '')
        # get the majority label
        count_list = clf.tree_.value[node_index, 0]
        max_index, max_value = max(enumerate(count_list), key=operator.itemgetter(1))
        max_label = labels[max_index]
        print(max_label)
        return tlevel_index
    
    else:
        # compute and print node label
        feature = features[clf.tree_.feature[node_index]]
        threshold = clf.tree_.threshold[node_index]
        print(kword, end=' ' if kword else '')
        print('if {} =< {}: '.format(feature, threshold))
        # recurse down the children
        left_index = clf.tree_.children_left[node_index]
        right_index = clf.tree_.children_right[node_index]
        ltlevel_index = _tree_rprint('then', clf, features, labels, left_index, tlevel_index+1)
        rtlevel_index = _tree_rprint('else', clf, features, labels, right_index, tlevel_index+1)
        # return the maximum depth of either one of the children
        return max(ltlevel_index,rtlevel_index)

