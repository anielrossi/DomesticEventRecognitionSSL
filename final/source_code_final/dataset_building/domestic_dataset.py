import json
import numpy as np
import copy
import matplotlib.pyplot as plt

FOLDER_DATA = '../json_final/input'

#### DEFINE CONSTRAINT HERE ###
MINLEN = 0.3  # duration
MAXLEN = 30.0
MIN_VOTES_CAT = 70  # minimum number of votes per category to produce a QE.
# maybe useless cause all have more than 72 votes (paper)

# MIN_HQ = 40  # minimum number of sounds with HQ labels per category. this was used for dump March09
MIN_HQ = 64  # minimum number of sounds with HQ labels per category, to include Bass drum but not others

# MIN_LQ = 75  # minimum number of sounds  with LQ labels per category
MIN_HQdev_LQ = 95  # minimum number of sounds between HQ and LQ labels per category
# goal is 90 ultimately, we put a bit more cause we will discar some more samples that bear more than one label

PERCENTAGE_DEV = 0.7 # split 70 / 30 for DEV / EVAL
# PERCENTAGE_DEV = 0.625 # split 62.5 / 27.5 for DEV / EVAL

# MIN_QE = 0.68  # minimum QE to accept the LQ as decent. this was used for dump March09
MIN_QE = 0.635  # minimum QE to accept the LQ as decent, to include Bass drum

FLAG_BARPLOT = False
FLAG_BOXPLOT = False
FLAG_BARPLOT_PARENT = False

"""load initial data with votes, clip duration and ontology--------------------------------- """
'''------------------------------------------------------------------------------------------'''

# this the result of the mapping from FS sounds to ASO.
# 268k sounds with basic metadata and their corresponding ASO id.
# useful to get the duration of every sound
try:
    with open(FOLDER_DATA + '/FS_sounds_ASO_postIQA.json') as data_file:
        data_duration = json.load(data_file)
except:
    raise Exception(
        'CHOOSE A MAPPING FILE AND ADD IT TO ' + FOLDER_DATA + 'json/ FOLDER (THE FILE INCLUDE DURATION INFORMATION NEEDED)')

# load json with votes, to select only PP and PNP
try:
    with open(FOLDER_DATA + '/votes_dumped_2018_May_16.json') as data_file:
        data_votes = json.load(data_file)
except:
    raise Exception(
        'ADD THE FILE CONTAINING THE VOTES (list of dict "value", "freesound_sound_id", "node_id") AND ADD IT TO THE FOLDER ' + FOLDER_DATA + 'json/')

# load json with ids and domestic categories info
try:
    with open(FOLDER_DATA + '/domestic_ids.json') as data_file:
        domestic_ids = json.load(data_file)
except:
    raise Exception('ADD AN ONTOLOGY JSON FILE TO THE FOLDER ' + FOLDER_DATA + 'json/')

# load json with ontology, to map aso_ids to understandable category names
try:
    with open(FOLDER_DATA + '/ontology.json') as data_file:
        data_onto = json.load(data_file)
except:
    raise Exception('ADD AN ONTOLOGY JSON FILE TO THE FOLDER ' + FOLDER_DATA + 'json/')

# load old json with votes in order to see the progress in time
# try:
#    with open(FOLDER_DATA + 'json/votes_dumped_2018_Jan_22.json') as data_file:
#        data_votes_old = json.load(data_file)
# except:
#    raise Exception(
#        'ADD THE FILE CONTAINING THE VOTES (list of dict "value", "freesound_sound_id", "node_id") AND ADD IT TO THE FOLDER ' + FOLDER_DATA + 'json/')
# data_onto is a list of dictionaries
# to retrieve them by id: for every dict o, we create another dict where key = o['id'] and value is o
data_onto_by_id = {o['id']: o for o in data_onto}


try:
    # load json with ontology, to map aso_ids to understandable category names
    # with open(FOLDER_DATA + 'json/votes_dumped_2018_Jan_22.json') as data_file:
    # with open(FOLDER_DATA + 'json/votes_dumped_2018_Feb_26.json') as data_file:
    # so far we were including in the data_votes_raw:
    # the trustable votes and the non trustable (verification clips not met)
    # from March1, we include only trustable
    # with open(FOLDER_DATA + 'json/votes_dumped_2018_Mar_01.json') as data_file:
    # with open(FOLDER_DATA + 'json/votes_dumped_2018_Mar_02.json') as data_file:
    # with open(FOLDER_DATA + 'json/votes_dumped_2018_Mar_09.json') as data_file:
    # with open(FOLDER_DATA + 'json/votes_dumped_2018_Mar_12.json') as data_file:
    with open(FOLDER_DATA + '/votes_dumped_2018_Mar_13.json') as data_file:
        data_votes_raw = json.load(data_file)
except:
    raise Exception('ADD A DUMP JSON FILE OF THE FSD VOTES TO THE FOLDER ' + FOLDER_DATA + 'json/')

# data_votes_raw is a dict where every key is a cat
# the value of every cat is a dict, that contains 5 keys: PP, PNP, NP, U, candidates
# the corresponding values are a list of Freesound ids


# -------------------------end of data reading--------------------------------------------------------
# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------


"""functions --------------------------------- """
'''------------------------------------------------------------------------------------------'''


def check_GT(group, fsid, catid, vote_groups, fsids_assigned_cat, data_sounds):
    # check if fsid has GT within a given group (PP,PNP,NP,U) of a category given by catid
    # if it does, add it to assigned fsids and send it to the corresponding group in data_sounds
    assigned = False
    if vote_groups[group].count(fsid) > 1:
        data_sounds[catid][group].append(fsid)
        fsids_assigned_cat.append(fsid)
        assigned = True
    return data_sounds, fsids_assigned_cat, assigned


def map_votedsound_2_disjointgroups_wo_agreement(fsid, catid, vote_groups, fsids_assigned_cat, data_sounds,
                                                 error_mapping_count_cat, count_risky_PP):
    # map the voted sound to a disjoint group  without inter-annotator agreement
    # using set of arbitrary rules that cover all possible options
    # being demanding now. only sending to PP when we are sure

    # retrieve votes for fsid.
    # we know there is only one in PP
    votes = []

    if fsid in vote_groups['PP']:
        votes.append(1.0)
    if fsid in vote_groups['PNP']:
        votes.append(0.5)
    if fsid in vote_groups['U']:
        votes.append(0.0)
    if fsid in vote_groups['NP']:
        votes.append(-1.0)

    # votes has all the votes for fsid. let us take decisions

    # trivial cases where there is only one single vote by one annotator
    if 1.0 in votes and 0.5 not in votes and -1.0 not in votes and 0.0 not in votes:
        # the only case where a sound is sent to PP without inter-annotator agreement
        data_sounds[catid]['PP'].append(fsid)
        fsids_assigned_cat.append(fsid)
    elif 1.0 not in votes and 0.5 in votes and -1.0 not in votes and 0.0 not in votes:
        # data_sounds[catid]['PNP'].append(fsid)
        # single vote of PNP may be a bit unreliable. safer to send it to U group
        # thus it goes to LQ (and not LQprior)
        data_sounds[catid]['U'].append(fsid)
        fsids_assigned_cat.append(fsid)
    elif 1.0 not in votes and 0.5 not in votes and -1.0 in votes and 0.0 not in votes:
        data_sounds[catid]['NP'].append(fsid)
        fsids_assigned_cat.append(fsid)
    elif 1.0 not in votes and 0.5 not in votes and -1.0 not in votes and 0.0 in votes:
        data_sounds[catid]['U'].append(fsid)
        fsids_assigned_cat.append(fsid)


    # rest of 11 cases. placing first the PP options in case there is some error.

    # 8 PP and PNP
    elif 1.0 in votes and 0.5 in votes and -1.0 not in votes and 0.0 not in votes:
        data_sounds[catid]['PNP'].append(fsid)
        fsids_assigned_cat.append(fsid)
        #candidate here
        count_risky_PP += 1
    # 9 PP and PNP and U
    elif 1.0 in votes and 0.5 in votes and -1.0 not in votes and 0.0 in votes:
        data_sounds[catid]['PNP'].append(fsid)
        fsids_assigned_cat.append(fsid)
        #candidate here
        count_risky_PP += 1

    # 1: NP and U
    elif 1.0 not in votes and 0.5 not in votes and -1.0 in votes and 0.0 in votes:
        data_sounds[catid]['NP'].append(fsid)
        fsids_assigned_cat.append(fsid)
    # 2: PNP and U
    elif 1.0 not in votes and 0.5 in votes and -1.0 not in votes and 0.0 in votes:
        data_sounds[catid]['U'].append(fsid)
        # candidate here
        fsids_assigned_cat.append(fsid)
    # 3: PNP and NP
    elif 1.0 not in votes and 0.5 in votes and -1.0 in votes and 0.0 not in votes:
        data_sounds[catid]['NP'].append(fsid)
        fsids_assigned_cat.append(fsid)
    # 4: PNP and NP and U
    elif 1.0 not in votes and 0.5 in votes and -1.0 in votes and 0.0 in votes:
        data_sounds[catid]['NP'].append(fsid)
        fsids_assigned_cat.append(fsid)


    # 5: PP and U
    elif 1.0 in votes and 0.5 not in votes and -1.0 not in votes and 0.0 in votes:
        data_sounds[catid]['U'].append(fsid)
        # candidate here
        fsids_assigned_cat.append(fsid)
    # 6: PP and NP
    elif 1.0 in votes and 0.5 not in votes and -1.0 in votes and 0.0 not in votes:
        data_sounds[catid]['U'].append(fsid)
        fsids_assigned_cat.append(fsid)
    # 7: PP and NP and U
    elif 1.0 in votes and 0.5 not in votes and -1.0 in votes and 0.0 in votes:
        data_sounds[catid]['U'].append(fsid)
        fsids_assigned_cat.append(fsid)

    # 10: PP and PNP and NP
    elif 1.0 in votes and 0.5 in votes and -1.0 in votes and 0.0 not in votes:
        data_sounds[catid]['U'].append(fsid)
        fsids_assigned_cat.append(fsid)
    # 11: PP and PNP and NP and U
    elif 1.0 in votes and 0.5 in votes and -1.0 in votes and 0.0 in votes:
        data_sounds[catid]['U'].append(fsid)
        fsids_assigned_cat.append(fsid)

    else:
        # print('\n something unexpetected happened in the mapping********************* \n')
        error_mapping_count_cat += 1
        # sys.exit('something unexpetected happened in the mapping!')

    return data_sounds, fsids_assigned_cat, error_mapping_count_cat, count_risky_PP



# Extract from all the votes only the domestic ones
domestic_votes = dict()
#votes of only the domestic categories (that I control from the external json file i.e. I can add or remove categories)
domestic_votes = {c: data_votes[c] for c in data_votes if c in domestic_ids.keys()}

""" # from data_votes to data_sounds ******************************************************************************"""
# Assign sounds to disjoint GROUPS (PP, PNP, NP, U) based on the combination of votes that they have

# create copy of data_votes
data_sounds = copy.deepcopy(domestic_votes)
for catid, vote_groups in data_sounds.items():
    data_sounds[catid]['PP'] = []
    data_sounds[catid]['PNP'] = []
    data_sounds[catid]['NP'] = []
    data_sounds[catid]['U'] = []
    data_sounds[catid]['QE'] = 0    # initialzed to 0. only if more than MIN_VOTES_CAT, we compute it

# count cases where the mapping from votes to sounds fails
error_mapping_count_cats = []

# to keep track of combinations
# PP + PNP and PP + PNP + U
count_risky_PP = 0

for catid, vote_groups in domestic_votes.items():
    # list to keep track of assigned fsids within a category, to achieve disjoint subsets of audio samples
    fsids_assigned_cat = []
    error_mapping_count_cat = 0

    # check GT in PP
    # check GT in the rest of the groups
    # if GT does not exist, take mapping decision without inter-annotator agreement
    for fsid in vote_groups['PP']:
        # print fsid
        # search for GT in this group
        if vote_groups['PP'].count(fsid) > 1:
            if fsid not in fsids_assigned_cat:
                data_sounds[catid]['PP'].append(fsid)
                fsids_assigned_cat.append(fsid)
        else:
            # search for GT in other groups of votes
            data_sounds, fsids_assigned_cat, assigned = check_GT('PNP', fsid, catid, vote_groups, fsids_assigned_cat,
                                                                 data_sounds)
            if not assigned:
                data_sounds, fsids_assigned_cat, assigned = check_GT('U', fsid, catid, vote_groups, fsids_assigned_cat,
                                                                     data_sounds)
            if not assigned:
                data_sounds, fsids_assigned_cat, assigned = check_GT('NP', fsid, catid, vote_groups, fsids_assigned_cat,
                                                                     data_sounds)

        # no GT was found for the annotation (2 votes in the same group).
        # we must take decisions without inter-annotator agreement

        if fsid not in fsids_assigned_cat:
            # map the voted sound to a disjoint group  without inter-annotator agreement
            data_sounds, fsids_assigned_cat, error_mapping_count_cat, count_risky_PP = map_votedsound_2_disjointgroups_wo_agreement(
                fsid, catid, vote_groups, fsids_assigned_cat, data_sounds, error_mapping_count_cat, count_risky_PP)

    # check GT in PNP
    # check GT in the remaining groups
    # if GT does not exist, take mapping decision without inter-annotator agreement
    for fsid in vote_groups['PNP']:
        # print fsid

        # only if the fsid has not been assigned in previous passes
        if fsid not in fsids_assigned_cat:
            # search for GT in this group
            if vote_groups['PNP'].count(fsid) > 1:
                if fsid not in fsids_assigned_cat:
                    data_sounds[catid]['PNP'].append(fsid)
                    fsids_assigned_cat.append(fsid)
            else:
                # search for GT in the remaining groups of votes
                data_sounds, fsids_assigned_cat, assigned = check_GT('U', fsid, catid, vote_groups, fsids_assigned_cat,
                                                                     data_sounds)
                if not assigned:
                    data_sounds, fsids_assigned_cat, assigned = check_GT('NP', fsid, catid, vote_groups,
                                                                         fsids_assigned_cat, data_sounds)

            # no GT was found for the annotation (2 votes in the same group).
            # we must take decisions without inter-annotator agreement

            if fsid not in fsids_assigned_cat:
                # map the voted sound to a disjoint group  without inter-annotator agreement
                data_sounds, fsids_assigned_cat, error_mapping_count_cat, count_risky_PP = map_votedsound_2_disjointgroups_wo_agreement(
                    fsid, catid, vote_groups, fsids_assigned_cat, data_sounds, error_mapping_count_cat, count_risky_PP)

    # check GT in U
    # check GT in the remaining groups
    # if GT does not exist, take mapping decision without inter-annotator agreement
    for fsid in vote_groups['U']:
        # print fsid

        # only if the fsid has not been assigned in previous passes
        if fsid not in fsids_assigned_cat:
            # search for GT in this group
            if vote_groups['U'].count(fsid) > 1:
                if fsid not in fsids_assigned_cat:
                    data_sounds[catid]['U'].append(fsid)
                    fsids_assigned_cat.append(fsid)
            else:
                # search for GT in the remaining groups of votes
                data_sounds, fsids_assigned_cat, assigned = check_GT('NP', fsid, catid, vote_groups, fsids_assigned_cat,
                                                                     data_sounds)

            # no GT was found for the annotation (2 votes in the same group).
            # we must take decisions without inter-annotator agreement

            if fsid not in fsids_assigned_cat:
                # map the voted sound to a disjoint group  without inter-annotator agreement
                data_sounds, fsids_assigned_cat, error_mapping_count_cat, count_risky_PP = map_votedsound_2_disjointgroups_wo_agreement(
                    fsid, catid, vote_groups, fsids_assigned_cat, data_sounds, error_mapping_count_cat, count_risky_PP)

    # check GT in NP
    # check GT in the remaining groups? no need to. already done in previous passes
    # if GT does not exist, take mapping decision without inter-annotator agreement
    for fsid in vote_groups['NP']:
        # print fsid

        # only if the fsid has not been assigned in previous passes
        if fsid not in fsids_assigned_cat:
            # search for GT in this group
            if vote_groups['NP'].count(fsid) > 1:
                if fsid not in fsids_assigned_cat:
                    data_sounds[catid]['NP'].append(fsid)
                    fsids_assigned_cat.append(fsid)
            # else: no need to. already done in previous passes

            # no GT was found for the annotation (2 votes in the same group).
            # we must take decisions without inter-annotator agreement
            else:
                # if fsid not in fsids_assigned_cat:
                # map the voted sound to a disjoint group  without inter-annotator agreement
                data_sounds, fsids_assigned_cat, error_mapping_count_cat, count_risky_PP = map_votedsound_2_disjointgroups_wo_agreement(
                    fsid, catid, vote_groups, fsids_assigned_cat, data_sounds, error_mapping_count_cat, count_risky_PP)

    # store mapping error for every category
    error_mapping_count_cats.append(error_mapping_count_cat)

    # for every category compute QE here number of votes len(PP) + len(PNP) / all
    # QE should only be computed if there are more than MIN_VOTES_CAT votes. else not reliable
    # if (len(vote_groups['PP']) + len(vote_groups['PNP']) + len(vote_groups['NP']) + len(
    #         vote_groups['U'])) >= MIN_VOTES_CAT:
    #     data_sounds[catid]['QE'] = (len(vote_groups['PP']) + len(vote_groups['PNP'])) / float(
    #         len(vote_groups['PP']) + len(vote_groups['PNP']) + len(vote_groups['NP']) + len(vote_groups['U']))

for i in domestic_ids:
    print(domestic_ids[i] + ": " + str(len(data_sounds[i]["PP"])))

#print("Potential risky PP (now in PNP): " + str(count_risky_PP) + ". They don't increase the possibility to add other categories")

dataset = dict()
for key,value in domestic_ids.items():
    dataset[value] = data_sounds[key]["PP"]

with open('../json_final/generated/dataset_categories.json', 'w') as fp:
    json.dump(dataset, fp)

print('\n')

final = dict()
for i in domestic_ids:
    if len(data_sounds[i]["PP"]) > 90:
        final[domestic_ids[i]] = data_sounds[i]["PP"]
        print(str(domestic_ids[i]) + ": " + str(len(data_sounds[i]["PP"])))

with open('../json_final/generated/dataset_categories_more_than_90.json', 'w') as fp:
    json.dump(final, fp)

# prefiltering number of chuncks of domestic sounds
# prefiltering = set()
# for i in domestic_ids:
#     prefiltering = prefiltering.union(set(domestic_votes[i]['PP']))
#     prefiltering = prefiltering.union(set(domestic_votes[i]['PNP']))
#     prefiltering = prefiltering.union(set(domestic_votes[i]['NP']))
#     prefiltering = prefiltering.union(set(domestic_votes[i]['U']))
#     prefiltering = prefiltering.union(set(domestic_votes[i]['candidates']))
#
# print("Number of chunks before filtering: " + str(len(prefiltering)))
# print("\n")
# print("Possible final dataset:\n")
#
# #HQ dataset without considering the number of sample per category
# middlefiltering = set()
# for i in domestic_ids:
#     middlefiltering = middlefiltering.union(set(data_sounds[i]['PP']))
#
# for i in final.keys():
#    print(i + ": " + str(len(final[i])))

#print("Number of chunks of the HQ dataset before balancing: " + str(len(middlefiltering)))
print("Two datasets are generated: dataset categories.json and dataset_categories_more_than_90.json")

"""DATASET CLOSED AND SAVED IN A JSON"""

