"""
modified code from https://github.com/minyoungg/wmigftl
what makes imagenet good for transfer learning - Minyoung Huh et al.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import nltk
from nltk.corpus import wordnet
import warnings
from utils.dataset_misc import *
nltk.download('wordnet')


def query_subclass_by_name(query_noun='dog', verbose=True):
    """ Given a query noun in string, finds all valid ImageNet wnid """
    try:
        query_wnid = wordnet.synsets(query_noun)[0]
    except Exception as e:
        if verbose:
            print(e)
        return []

    valid_wnids = []
    for wnid in IMAGENET_WNID_TO_LABEL.keys():
        s = wnid_to_synset(str(wnid))
        if is_hyponym(s, query_wnid):
            valid_wnids.append(int(wnid))

    if len(valid_wnids) == 0:
        if verbose:
            warnings.warn('no wnid found for the query: {}'.format(query_noun))
    return np.sort(valid_wnids)


def get_parent_wnid(wnid):
    """ Given wnid get parent wnid """
    return 'n' + str(synset_to_wnid(wnid_to_synset(wnid).hypernyms()[0])).zfill(8)


def synset_to_wnid(synset):
    """ Converts synset to wnid. Synset is a wordnet node object"""
    return synset.offset()


def wnid_to_synset(wnid):
    """ Converts wnid back into synset (only nouns) """
    if type(wnid) is str:
        if wnid[0] == 'n':
            wnid = int(wnid[1:])
        else:
            wnid = int(wnid)
    return wordnet._synset_from_pos_and_offset('n', wnid)


def wnid_str_to_int(str_wnid):
    """ string wnid to integer wnid"""
    return int(str_wnid[1:].lstrip('0'))


def wnid_to_noun(wnid):
    """ Converts wnid to noun (chooses the first definition)"""
    return wnid_to_synset(wnid).lemmas()[0].name().replace('_', ' ')


def is_hyponym(syn1, syn2):
    """ Checks if syn1 is a child of syn2 """
    while syn1 != syn2:
        hypernyms = syn1.hypernyms()
        if len(hypernyms) == 0:
            return False
        syn1 = hypernyms[0]
    return True


def wnid_depth(wnid):
    """ Computes the depth of the given wnid """
    syn = wnid_to_synset(wnid)
    depth = 0
    hyper = syn.hypernyms()
    while len(hyper) != 0:
        depth += 1
        # move up, choose the first parent. Sometimes there are more than 1 parent.
        syn = syn.hypernyms()[0]
        hyper = syn.hypernyms()
    return depth


def read_synset_file(synset_words_path):
    """
    Reads synset.txt or synset_words.txt
    Returns
        wnid_array - list of wnid strings
    """
    wnid_array = [line.split(' ')[0] for line in open(synset_words_path, 'r')]
    return wnid_array


def read_txt_file(txt_file):
    """
    Reads imagenet train.txt and val.txt files
    """
    return [line for line in open(txt_file, 'r')]


def wnid_statistics(wnid_arr):
    """
    Computes some simple statistics on a list of wnid
    Args
        wnid_arr - An array of wnids
    Returns
        stats - Dictionary which summarizes the computed statistics
    """
    stats = {}
    depth_arr = [wnid_depth(w) for w in wnid_arr]
    stats['depth_arr'] = depth_arr
    stats['min_depth'], stats['max_depth'] = np.min(depth_arr), np.max(depth_arr)
    return stats


def get_coco_valid_wnids():
    wnids = {}
    for n in COCO_INSTANCE_CATEGORY_NAMES:
        v = query_subclass_by_name(n, verbose=False)
        if len(v) != 0:
            wnids[n] = v
    return wnids


def get_pascal_valid_wnids():
    wnids = {}
    for n in PASCAL_INSTANCE_CATEGORY_NAMES:
        v = query_subclass_by_name(n, verbose=False)
        if len(v) != 0:
            wnids[n] = v
    return wnids


def to_onehot(lbls, num_classes=1000):
    """ converts list of labels into  onehot encoding """
    c = np.zeros((len(lbls), num_classes))
    for i, l in enumerate(lbls):
        c[i, l] = 1
    return np.array(c)
