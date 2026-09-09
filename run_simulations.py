import os
import random
import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp

from get_summary import *
from orth_neighborhood_utils import *
from predictive_coding_model import *
from create_stimulus_counterbalanced_lists import *

import os
import random
import numpy as np
import pandas as pd

os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.makedirs("data", exist_ok=True)
random.seed(1)
np.random.seed(1)


def run_simulation(**kwargs):
    # run a simulation and keep only the components needed for plotting and data analysis (to avoid out-of-memory errors)
    full_simulation = Simulation(**kwargs)
    data,fname,sim_input_bottomup,sim_input_topdown = full_simulation.simulation_data, full_simulation.sim_filename, full_simulation.sim_input_bottomup, full_simulation.sim_input_topdown
    del full_simulation
    return {"simulation_data" : data,
             "sim_filename" : fname,
             "sim_input_bottomup": sim_input_bottomup,
             "sim_input_topdown": sim_input_topdown}
    # return full_simulation

lexicon = Lexicon()
NUM_ITERS = 20
THRESHOLD = 2.8

wordlist = np.array(lexicon.words)

##################### DEFINE STIMULI FOR ALL CONDITIONS ##################### 
# generate_stims_for_all_conditions(lexicon)

quads = find_counterbalanced_quads(lexicon,min_sem_overlap = 2)
stim_dict = create_stimulus_lists(quads, lexicon)
# verify_stimulus_lists(stim_dict, lexicon)
# pseudowords = generate_matched_pseudowords(stim_dict['standard_words'], lexicon)
# np.save('./helper_txt_files/standard_stims_word_idx.npy', stim_dict['standard_idx'])
# np.save('./helper_txt_files/sem_related_word_idx_nonneighbor.npy', stim_dict['sem_related_idx'])
# np.save('./helper_txt_files/fully_unrelated_word_idx_1.npy', stim_dict['unrelated_idx'])
# np.save('./helper_txt_files/pseudowords.npy', np.array(pseudowords))

##### 

standard_stims_word_idx  = stim_dict['standard_idx']
fully_unrelated_word_idx_1  = stim_dict['unrelated_idx']
sem_related_word_idx_nonneighbor  = stim_dict['sem_related_idx']


standard_stims = wordlist[standard_stims_word_idx]
unrelated_stims = wordlist[fully_unrelated_word_idx_1]
semrelated_stims = wordlist[sem_related_word_idx_nonneighbor]

final_standard_idx = [list(lexicon.words).index(w) for w in standard_stims]
final_unrelated_idx = [list(lexicon.words).index(w) for w in unrelated_stims]
final_semrelated_idx = [list(lexicon.words).index(w) for w in semrelated_stims]

shared_feats = lexicon.semfeatmatrix[:,final_standard_idx] * lexicon.semfeatmatrix[:,final_semrelated_idx] 
assert (lexicon.semfeatmatrix[:,final_unrelated_idx] * lexicon.semfeatmatrix[:,final_semrelated_idx]).sum() == 0

num_stims = standard_stims.shape[0]

##################### PSEUDOWORD SIMULATIONS #####################
# pseudowords = np.load('./helper_txt_files/pseudowords.npy')
# pseudoword_stims = np.array(pseudowords)[np.array(pseudowords) != None]
# word_stims = np.array(standard_stims)[np.array(pseudowords) != None]

# pseudoword_simulation = run_simulation(sim_input_bottomup = pseudoword_stims, clamp_iterations = NUM_ITERS, sim_filename = 'pseudoword_simulation')
# word_simulation = run_simulation(sim_input_bottomup = word_stims, clamp_iterations = NUM_ITERS, sim_filename = 'word_simulation')
# pseudoword_crossers = (np.max(pseudoword_simulation['simulation_data']['max_lex_state_activation'][0,:,:], axis = -1) > THRESHOLD)
# word_crossers = (np.max(word_simulation['simulation_data']['max_lex_state_activation'][0,:,:], axis = -1) > THRESHOLD)
# pseudoword_crossers_count = pseudoword_crossers.sum()
# word_crossers_count = word_crossers.sum()

# print(f'Percentage of words that crossed the threshold is {word_crossers_count/word_crossers.shape[0]*100}%')
# print(f'Percentage of pseudowords that did not cross the threshold is {(1 - pseudoword_crossers_count/pseudoword_crossers.shape[0])*100}%')

# ##################### CONTEXTUAL EFFECT SIMULATIONS #####################
standard_simulation = run_simulation(sim_input_bottomup = standard_stims[:num_stims], clamp_iterations = NUM_ITERS, sim_filename = 'standard_simulation_090826')

# Run the semantic priming simulation
sem_priming = {'semunrelated': run_simulation(sim_input_bottomup = unrelated_stims[:num_stims], clamp_iterations =NUM_ITERS, blanks_before_clamp = 5, prevSim = standard_simulation, sim_filename = 'sem_priming_semunrelated5_090826'),
                'semrelated': run_simulation(sim_input_bottomup = semrelated_stims[:num_stims], clamp_iterations =NUM_ITERS, blanks_before_clamp = 5, prevSim = standard_simulation, sim_filename = 'sem_priming_semrelated5_090826')}

# Run the repetition priming simulation
rep_priming = {'unrepeated': run_simulation(sim_input_bottomup = unrelated_stims[:num_stims], clamp_iterations =NUM_ITERS, blanks_before_clamp = 5, prevSim = standard_simulation, sim_filename = 'rep_priming_unrepeated5_090826'),
               'repeated' : run_simulation(sim_input_bottomup = standard_stims[:num_stims], clamp_iterations =NUM_ITERS, blanks_before_clamp = 5, prevSim = standard_simulation, sim_filename = 'rep_priming_repeated5_090826')}


# Run the lexical predictability simulation
cloze_levels = {
    "low_cloze": 1/lexicon.size, 
                "med_low_cloze": 0.25,
                "med_high_cloze": 0.5,
                "high_cloze": 0.90,
                }

cloze_simulations_preactivate = {}
cloze_simulations_bottomup = {}

for key, val in cloze_levels.items():
    # pre-activate each of the 512 standard inputs from the top down
    cloze_simulations_preactivate.update({key: run_simulation(sim_input_topdown = standard_stims[:num_stims], 
                                                              clamp_iterations =NUM_ITERS,
                                                              BU_TD_mode = "top_down", 
                                                              cloze_target = val, 
                                                              sim_filename = f'cloze_simulations_preact_{key}_090826')})
    # present each of the 512 standard inputs from the bottom up
    cloze_simulations_bottomup.update({key: run_simulation(sim_input_bottomup = standard_stims[:num_stims], 
                                                           sim_input_topdown = standard_stims[:num_stims],
                                                           cloze_target = val,
                                                           clamp_iterations =NUM_ITERS,
                                                           BU_TD_mode = "bottom_up", 
                                                           prevSim = cloze_simulations_preactivate[key], 
                                                           sim_filename = f'cloze_simulations_bottomup{key}_090826')})


# # faster if the data is pre-saved:
# for key, val in cloze_levels.items():
#     cloze_simulations_preactivate.update({key: run_simulation(sim_input = standard_stims, clamp_iterations =NUM_ITERS,BU_TD_mode = "top_down", cloze_target = val, sim_filename = f'cloze_simulations_preact_{key}_090826')})
#     cloze_simulations_bottomup.update({key: run_simulation(sim_filename = f'cloze_simulations_bottomup{key}_090826')})

# Run the lexical prediction violation simulation
lexical_violation = {"low_constraint_unexpected": run_simulation(sim_input_bottomup = unrelated_stims[:num_stims], 
                                                                 sim_input_topdown = standard_stims[:num_stims], 
                                                                 clamp_iterations =NUM_ITERS,
                                                                 BU_TD_mode = "bottom_up", 
                                                                 cloze_target = 1/lexicon.size,
                                                                 prevSim = cloze_simulations_preactivate["low_cloze"], 
                                                                 sim_filename = 'lexviol_LCunexp_090826'), 
                    "high_constraint_unexpected": run_simulation(sim_input_bottomup = unrelated_stims[:num_stims], 
                                                                 sim_input_topdown = standard_stims[:num_stims], 
                                                                 clamp_iterations =NUM_ITERS,
                                                                 BU_TD_mode = "bottom_up", 
                                                                 cloze_target = 0.90,
                                                                 prevSim = cloze_simulations_preactivate["high_cloze"], 
                                                                 sim_filename = 'lexviol_HCunexp_090826'),
                    "high_constraint_expected": run_simulation(sim_filename = f'cloze_simulations_bottomuphigh_cloze_090826')
                    }

# # if presaved:
# lexical_violation = {"low_constraint_unexpected": run_simulation(sim_filename = 'lexviol_LCunexp_090826'), 
#                     "high_constraint_unexpected": run_simulation(sim_filename = 'lexviol_HCunexp_090826'),
#                     "high_constraint_expected": run_simulation(sim_filename = f'cloze_simulations_bottomuphigh_cloze_090826')
#                     }


semantic_prediction_overlap = {"semunrelated_90cloze": run_simulation(sim_input_bottomup = unrelated_stims[:num_stims], 
                                                                      sim_input_topdown = standard_stims[:num_stims], 
                                                                      BU_TD_mode = "bottom_up",
                                                                      cloze_target = 0.9,
                                                                      clamp_iterations =NUM_ITERS, 
                                                                      prevSim = cloze_simulations_preactivate["high_cloze"], 
                                                                      sim_filename = 'sempredoverlap_semunrelated_90cloze_090826'),
                                "semrelated_90cloze": run_simulation(sim_input_bottomup = semrelated_stims[:num_stims], 
                                                                     sim_input_topdown = standard_stims[:num_stims], 
                                                                     BU_TD_mode = "bottom_up",
                                                                     cloze_target = 0.9,
                                                                     clamp_iterations =NUM_ITERS, 
                                                                     prevSim = cloze_simulations_preactivate["high_cloze"], 
                                                                     sim_filename = 'sempredoverlap_semrelated_90cloze_090826')}

semantic_prediction_overlap = {}
semantic_prediction_overlap.update({"semunrelated_90cloze": run_simulation(sim_filename = f'sempredoverlap_semunrelated_90cloze_090826')})
semantic_prediction_overlap.update({"semrelated_90cloze": run_simulation(sim_filename = f'sempredoverlap_semrelated_90cloze_090826')})
semantic_prediction_overlap.update({"high_constraint_expected": run_simulation(sim_filename = f'cloze_simulations_bottomuphigh_cloze_090826')})
# 03-08-2026 MERGE THIS WITH THE LEXVIOL SIMULATION AND THEN PLOT THEM TOGETHER

# four conditions: expected vanilla 70-70, 30-30
# withcomp: comprel70/target30 target30 or comp_unrel70/target30

configs = [
    dict(
        BU_TD_mode="top_down",
        sim_input_topdown=standard_stims[:num_stims],
        cloze_target=0.3,
        clamp_iterations=20,
    ), # vanilla 30-30?
    dict(
        BU_TD_mode="top_down",
        sim_input_topdown=standard_stims[:num_stims],
        sim_input_competitor=unrelated_stims[:num_stims],
        cloze_target=0.3,
        cloze_competitor=0.7,
        clamp_iterations=20,
    ), # compunrel70/target30
    dict(
        BU_TD_mode="top_down",
        sim_input_topdown=standard_stims[:num_stims],
        sim_input_competitor=semrelated_stims[:num_stims],
        cloze_target=0.3,
        cloze_competitor=0.7,
        clamp_iterations=20,
    ), #comprel70/target30
    dict(
        BU_TD_mode="top_down",
        sim_input_topdown=standard_stims[:num_stims],
        cloze_target=0.7,
        clamp_iterations=20,
    ), #vanilla  70-70
]

comp_simnames = ['no_comp', 'unrel_comp', 'rel_comp', 'expected']
comp_sims = {}

for condition, condition_config in zip(comp_simnames, configs):

    # Run the top-down preactivation phase
    preactsim = run_simulation(
        **condition_config,
        sim_filename=f'competition_preact_{condition}_090826'
    )

    # Then present the target bottom-up
    comp_sims[condition] = run_simulation(
        BU_TD_mode="bottom_up",
        sim_input_bottomup=standard_stims[:num_stims],
        prevSim=preactsim,
        clamp_iterations=20,
        sim_filename=f'competition_bottomup_{condition}_090826'
    )
    del preactsim

##################### PLOT DATA FROM ALL CONTEXTUAL SIMULATIONS #####################

all_simulations = [sem_priming, rep_priming, cloze_simulations_bottomup, lexical_violation, semantic_prediction_overlap, comp_sims]
all_simulations_names = ["sem_priming", "rep_priming", "cloze_simulations_bottomup", "lexical_violation", "semantic_prediction_overlap", "competitor_simulation"]

def simulation_accuracy_info(the_simulation, num_iters_target_presentation=20):
    crossers = []
    for condition in the_simulation.keys():
        # retrieve the activity of the most active lexical state for each trial, ie target input, at all iterations
        most_active_state_activity_per_trial = the_simulation[condition]['simulation_data']['max_lex_state_activation'][0][:,-num_iters_target_presentation:] #(num_trials,num_iterations)
        # indicate whether the threshold was crossed for each trial
        threshold_was_crossed = np.any(most_active_state_activity_per_trial > THRESHOLD,axis = 1)
        # for each trial, find the iteration at which the threshold was crossed
        threshold_crossing_iteration_per_trial = np.argmax(most_active_state_activity_per_trial > THRESHOLD,axis = 1) [threshold_was_crossed]#(threshold_crossing_trials,20)
        # for each trial, find the identity of the lexical state that crossed the threshold
        most_active_state_identity_per_trial = the_simulation[condition]['simulation_data']['max_lex_state_identity'][0][threshold_was_crossed,-num_iters_target_presentation:]#(threshold_crossing_trials,20)
        identity_of_threshold_crosser = most_active_state_identity_per_trial[np.arange(np.sum(threshold_was_crossed)), threshold_crossing_iteration_per_trial] #(threshold_crossing_trials,)
        # retrieve the identity of the "correct" target that should have crossed the threshold
        target_identity = np.array([list(lexicon.words).index(w) for w in np.array(the_simulation[condition]['sim_input_bottomup'])[threshold_was_crossed]])
        # check how many of the lexical states that crossed the threshold matched the correct target state
        print(f'The number of threshold crossers in {the_simulation[condition]["sim_filename"]} that matched the identity was {np.sum(target_identity == identity_of_threshold_crosser)} out of {threshold_was_crossed.shape[0]}, {(np.sum(target_identity == identity_of_threshold_crosser)/threshold_was_crossed.shape[0])*100}%')
        crossers.append(target_identity == identity_of_threshold_crosser)
    return crossers



def get_threshold_crossing_info(the_simulation, condition, threshold,num_iters_target_presentation = NUM_ITERS, return_iteration_idx = False):

    # retrieve the correct, target identity
    target_identity = np.array([list(lexicon.words).index(w) for w in the_simulation[condition]['sim_input_bottomup']])
    # next, retrieve the activity of the most active lexical state for each trial, ie target input, at all iterations
    most_active_state_activity_per_trial = the_simulation[condition]['simulation_data']['max_lex_state_activation'][0][:,-num_iters_target_presentation:] #(num_trials,num_iterations)
    most_active_state_identity_per_trial = the_simulation[condition]['simulation_data']['max_lex_state_identity'][0][:,-num_iters_target_presentation:] #(num_trials,num_iterations)
    # initialize 3 vectors, indicating whether the correct item crossed the threshold and what the index-identity and word-identity of the threshold crosser was
    correct_item_identified = np.ones(the_simulation[condition]['sim_input_bottomup'].shape[0])*np.nan
    threshold_crossing_iteration_ = np.ones(the_simulation[condition]['sim_input_bottomup'].shape[0])*np.nan
    idx_identity_of_threshold_crosser = (np.ones(the_simulation[condition]['sim_input_bottomup'].shape[0])*np.nan).astype('int16')
    word_identity_of_threshold_crosser = (np.ones(the_simulation[condition]['sim_input_bottomup'].shape[0])*np.nan).astype('<U4')

    # indicate whether the threshold was crossed for each trial
    threshold_was_crossed = np.any(most_active_state_activity_per_trial > threshold,axis = 1)

    for trial_idx,word in enumerate(the_simulation[condition]['sim_input_bottomup']):
        if threshold_was_crossed[trial_idx]:
            threshold_crossing_iteration = np.argmax(most_active_state_activity_per_trial[trial_idx] > threshold)
            threshold_crossing_iteration_[trial_idx] = threshold_crossing_iteration
            idx_identity_of_threshold_crosser[trial_idx] = most_active_state_identity_per_trial[trial_idx, threshold_crossing_iteration] 
            correct_item_identified[trial_idx] = idx_identity_of_threshold_crosser[trial_idx] == target_identity[trial_idx]

    word_identity_of_threshold_crosser[~np.isnan(idx_identity_of_threshold_crosser)] = lexicon.words[idx_identity_of_threshold_crosser[~np.isnan(idx_identity_of_threshold_crosser)].astype('int32')]
    if return_iteration_idx:
        return threshold_was_crossed, correct_item_identified, word_identity_of_threshold_crosser, threshold_crossing_iteration_
    return threshold_was_crossed, correct_item_identified, word_identity_of_threshold_crosser

merged_sempredoverlap_noconstraint = {}
for cond in ['low_constraint_unexpected','high_constraint_unexpected', 'semrelated_90cloze', 'high_constraint_expected']:
    merged_sempredoverlap_noconstraint[cond] = {}
    merged_sempredoverlap_noconstraint[cond]['simulation_data'] = {}

for cond in ['low_constraint_unexpected','high_constraint_unexpected','high_constraint_expected']:
    for var in ['total_lexsem_err', 'max_lex_state_activation']:
        merged_sempredoverlap_noconstraint[cond]['simulation_data'][var] = lexical_violation[cond]['simulation_data'][var]

for var in ['total_lexsem_err', 'max_lex_state_activation']:
    merged_sempredoverlap_noconstraint['semrelated_90cloze']['simulation_data'][var] = semantic_prediction_overlap['semrelated_90cloze']['simulation_data'][var]


priming_sims = {}
for cond in ['semunrelated', 'semrelated','repeated']:
    priming_sims[cond] = {}
    priming_sims[cond]['simulation_data'] = {}

for cond in ['semunrelated','semrelated']:
    for var in ['total_lexsem_err', 'max_lex_state_activation']:
        priming_sims[cond]['simulation_data'][var] = sem_priming[cond]['simulation_data'][var]

for var in ['total_lexsem_err', 'max_lex_state_activation']:
    priming_sims['repeated']['simulation_data'][var] = rep_priming['repeated']['simulation_data'][var]


all_simulations = [priming_sims, cloze_simulations_bottomup, merged_sempredoverlap_noconstraint, comp_sims]
all_simulations_names = ["priming_sims", "cloze_simulations_bottomup", "merged_sempredoverlap_noconstraint", "competitor_simulation"]


tci_dic = {}
for sim, simname in zip(all_simulations, all_simulations_names):
    tci_dic[simname] = {}
    for subsim_name in sim.keys():
        threshold_crossing_iterations = []
        for a_threshold in np.arange(0.01,4.01,0.01):
            activity = sim[subsim_name]['simulation_data']['max_lex_state_activation'][0][:, -NUM_ITERS:]
            boolean_threshold_crossing = activity > a_threshold
            
            threshold_was_crossed = np.any(boolean_threshold_crossing, axis=1)  # (num_trials,)
            
            tci = np.full(activity.shape[0], np.nan)  # default to NaN, not 0
            tci[threshold_was_crossed] = np.argmax(boolean_threshold_crossing[threshold_was_crossed], axis=1)
            
            threshold_crossing_iterations.append(tci)
        tci_dic[simname][subsim_name] = threshold_crossing_iterations


os.makedirs("./plots/threshold_plots", exist_ok=True)

# ---- USER CONTROL ----
sim_order = ['priming_sims', 'cloze_simulations_bottomup', 'merged_sempredoverlap_noconstraint', 'competitor_simulation']
panel_titles = ['(a) Priming', '(b) Lexical Probability', '(c) Lex. Prediction Violation', '(d) More Probable Competitor']


legend_rename = {
    "priming_sims": {
        'semunrelated': 'Unrelated',
        'semrelated': 'Semantically Related',
        'repeated': 'Repeated',
    },
    "cloze_simulations_bottomup": {
        'low_cloze': '.1%',
        'med_low_cloze': '25%',
        'med_high_cloze': '50%',
        'high_cloze': '90%',
    },
    "merged_sempredoverlap_noconstraint": {
        'low_constraint_unexpected': 'Low Constraint Unexp. Unrel.',
        'high_constraint_unexpected': 'High Constraint Unexp. Unrel',
        'semrelated_90cloze': 'High Constraint Unexp. Related',
        'high_constraint_expected': 'High Constraint Expected',
    },
    "competitor_simulation": {
        'no_comp' : 'Less Expected without Competitor',
        'unrel_comp' : 'Less Expected with Unrelated Competitor',
        'rel_comp' : 'Less Expected with Related Competitor',
        'expected' : f'More Expected',
    },
}

#############################

# -----------------------
# STYLE DICTIONARY
# -----------------------
condition_styles = {
    "priming_sims": {
        "semunrelated": "k:",
        "semrelated": "k--",
        "repeated": "k-",
    },
    "competitor_simulation": {
        "no_comp": "b:",
        "unrel_comp": "C1--",
        "rel_comp": "k--",
        "expected": "k-",
    },
    "cloze_simulations_bottomup": {
        "high_cloze": "k-",
        "med_high_cloze": "k--",
        "med_low_cloze": "k-.",
        "low_cloze": "k:",
    },
    "merged_sempredoverlap_noconstraint": {
        "low_constraint_unexpected": "k:",
        "high_constraint_unexpected": "r--",
        "semrelated_90cloze": "k--",
        "high_constraint_expected": "k-",
    }
}

# -----------------------
# PLOTTING
# -----------------------
thresholds = np.arange(0.01, 4.01, 0.01)

fig, axes = plt.subplots(
    2, 2,
    figsize=(11, 6),
    sharex=True,
    sharey=True
)

axes = axes.flatten()

for ax, simname, title in zip(axes, sim_order, panel_titles):

    subsims = tci_dic[simname]

    for subsim_name, arr_list in subsims.items():

        arr = np.array(arr_list)

        means = np.nanmean(arr, axis=1)
        sems = (
            np.nanstd(arr, axis=1)
            / np.sqrt(np.sum(~np.isnan(arr), axis=1))
        )

        style = condition_styles.get(
            simname, {}
        ).get(
            subsim_name, "k-"
        )

        line = ax.plot(
            thresholds,
            means,
            style,
            linewidth=3,
            label=legend_rename.get(
                simname, {}
            ).get(
                subsim_name,
                subsim_name
            ),
        )[0]

    # X-axis formatting
    ax.set_xlim(0, 4)
    ax.set_xticks(np.arange(0, 4.01, 0.5))

    ax.set_title(title, fontsize=10)
    ax.legend(
        fontsize=10,
        frameon=False
    )

# Hide unused panels
for ax in axes[len(sim_order):]:
    ax.axis("off")

# Axis labels
axes[0].set_ylabel("Mean crossing\niteration")
axes[2].set_ylabel("Mean crossing\niteration")

axes[2].set_xlabel("Threshold")
axes[3].set_xlabel("Threshold")

fig.tight_layout()

fig.savefig(
    "./plots/threshold_plots/Figure11_all_simulations_thresholds.png",
    dpi=300,
    bbox_inches="tight"
)

fig.savefig(
    "./plots/threshold_plots/Figure11_all_simulations_thresholds.svg",
    dpi=300,
    bbox_inches="tight"
)
plt.close(fig)



def get_max_yval(all_simulations):
    max_yval_st = 0
    max_yval_err = 0
    for simulation in all_simulations:
        for sub_sim in simulation.keys():
            data_to_plot_st = np.mean(simulation[sub_sim]['simulation_data']['max_lex_state_activation'][0],axis = 0).T
            data_to_plot_err = np.mean(simulation[sub_sim]['simulation_data']['total_lexsem_err'][0],axis = 0).T
            if max(data_to_plot_st) >= max_yval_st:
                max_yval_st = max(data_to_plot_st)
            if max(data_to_plot_err) >= max_yval_err:
                max_yval_err = max(data_to_plot_err)
    return max_yval_st,max_yval_err

merged_sempredoverlap_noconstraint = {}
for cond in ['low_constraint_unexpected','high_constraint_unexpected', 'semrelated_90cloze', 'high_constraint_expected']:
    merged_sempredoverlap_noconstraint[cond] = {}
    merged_sempredoverlap_noconstraint[cond]['simulation_data'] = {}

for cond in ['low_constraint_unexpected','high_constraint_unexpected','high_constraint_expected']:
    for var in ['total_lexsem_err', 'max_lex_state_activation']:
        merged_sempredoverlap_noconstraint[cond]['simulation_data'][var] = lexical_violation[cond]['simulation_data'][var]

for var in ['total_lexsem_err', 'max_lex_state_activation']:
    merged_sempredoverlap_noconstraint['semrelated_90cloze']['simulation_data'][var] = semantic_prediction_overlap['semrelated_90cloze']['simulation_data'][var]


##################### CREATE CSV DATA FROM ALL SIMULATIONS #####################
THRESHOLD = 2.8
def create_simulation_df(simulation, conditions_dict,simulation_name):
    # given a simulation and a conditions_dict, write a CSV file with the right columns
    sub_simulations = list(simulation.keys()) # list of conditions (e.g., unrepeated, repeated)
    number_of_conditions = len(sub_simulations)
    assert conditions_dict['factors'].shape[0] == number_of_conditions # each condition must have a name
    num_of_trials_condition1 = simulation[sub_simulations[0]]['simulation_data']['total_lexsem_err'].shape[1]
    num_iters_condition1 = simulation[sub_simulations[0]]['simulation_data']['total_lexsem_err'].shape[-1]
    for subsim in sub_simulations[1:]:
        num_of_trials_conditionK = simulation[subsim]['simulation_data']['total_lexsem_err'].shape[1]
        num_iters_conditionK = simulation[subsim]['simulation_data']['total_lexsem_err'].shape[-1]
        assert num_of_trials_conditionK == num_of_trials_condition1 # make sure all conditions have an equal number of trials
        assert num_iters_conditionK == num_iters_condition1 # make sure all conditions have an equal number of iterations per trial
    num_trials_per_condition = num_of_trials_condition1
    num_iters_per_trial = num_iters_condition1
    # get a 10-iteration time_window covering iterations 2 through 11, inclusive.
    # Note that -20 is the first iteration of the final word. -20 + 1 is the 2nd iteration; -20 + 11 is the 12th iteration. 
    time_window = np.arange(num_iters_per_trial)[-NUM_ITERS + 1:-NUM_ITERS + 11] 
    # get a four-iteration window covering iterations 2 to 6 inclusive
    # i.e. iterations 2 to 7 instead:
    time_window = np.arange(num_iters_per_trial)[-NUM_ITERS + 1:-NUM_ITERS + 6] 

    WordInds_LexSemErr_ThresholdCrossingIteration_ONsize_Frequency_NumSemFeats_ThresholdWasCrossed_CorrectItemCrossed_WordThatCrossed = np.zeros((num_trials_per_condition*number_of_conditions,9)).astype('object')
    sim_input_all = []

    for condition_number, subsim in enumerate(sub_simulations):
        # retrieve start and end inds in the data matrix
        # confirm that these are the right ones.
        start_ind = num_trials_per_condition*condition_number 
        end_ind = num_trials_per_condition*(condition_number +1)

        sim_input_inds = np.array(get_correct_inds(simulation[subsim]['sim_input_bottomup'] ,lexicon.words))
        sim_input_all.extend(simulation[subsim]['sim_input_bottomup'])
        LexSemErr_FullTimeCourse = simulation[subsim]['simulation_data']['total_lexsem_err'][0]
        
        mean_LexSemErr = np.mean(LexSemErr_FullTimeCourse[:,time_window], axis = 1)


        WordInds_LexSemErr_ThresholdCrossingIteration_ONsize_Frequency_NumSemFeats_ThresholdWasCrossed_CorrectItemCrossed_WordThatCrossed[start_ind:end_ind,1] = mean_LexSemErr

        boolean_threshold_crossing = simulation[subsim]['simulation_data']['max_lex_state_activation'][0][:,-NUM_ITERS:] > THRESHOLD
        threshold_crossing_iteration = np.argmax(boolean_threshold_crossing, axis = 1)
        WordInds_LexSemErr_ThresholdCrossingIteration_ONsize_Frequency_NumSemFeats_ThresholdWasCrossed_CorrectItemCrossed_WordThatCrossed[start_ind:end_ind,2] = threshold_crossing_iteration
        WordInds_LexSemErr_ThresholdCrossingIteration_ONsize_Frequency_NumSemFeats_ThresholdWasCrossed_CorrectItemCrossed_WordThatCrossed[start_ind:end_ind,6][boolean_threshold_crossing[:,-1]] = 1
        if not np.any(np.isnan(sim_input_inds)):
            WordInds_LexSemErr_ThresholdCrossingIteration_ONsize_Frequency_NumSemFeats_ThresholdWasCrossed_CorrectItemCrossed_WordThatCrossed[start_ind:end_ind,0] = sim_input_inds
            WordInds_LexSemErr_ThresholdCrossingIteration_ONsize_Frequency_NumSemFeats_ThresholdWasCrossed_CorrectItemCrossed_WordThatCrossed[start_ind:end_ind,3] = lexicon.ONsize.T[sim_input_inds][:,0]
            WordInds_LexSemErr_ThresholdCrossingIteration_ONsize_Frequency_NumSemFeats_ThresholdWasCrossed_CorrectItemCrossed_WordThatCrossed[start_ind:end_ind,4] = lexicon.frequency.T[sim_input_inds][:,0]
            # conc = np.array([lexicon.concreteness]).T
            # conc[conc == 0] = -1 # code nonrich items as -1
            WordInds_LexSemErr_ThresholdCrossingIteration_ONsize_Frequency_NumSemFeats_ThresholdWasCrossed_CorrectItemCrossed_WordThatCrossed[start_ind:end_ind,5] = lexicon.num_semfeats[sim_input_inds]
            ThresholdWasCrossed,CorrectItemCrossed,WordThatCrossed = get_threshold_crossing_info(simulation, subsim, THRESHOLD)
            WordInds_LexSemErr_ThresholdCrossingIteration_ONsize_Frequency_NumSemFeats_ThresholdWasCrossed_CorrectItemCrossed_WordThatCrossed[start_ind:end_ind,6] = ThresholdWasCrossed # binary value indicating if any item crossed the threshold
            WordInds_LexSemErr_ThresholdCrossingIteration_ONsize_Frequency_NumSemFeats_ThresholdWasCrossed_CorrectItemCrossed_WordThatCrossed[start_ind:end_ind,7] = CorrectItemCrossed # binary value indicating if the correct item crossed the threshold
            WordInds_LexSemErr_ThresholdCrossingIteration_ONsize_Frequency_NumSemFeats_ThresholdWasCrossed_CorrectItemCrossed_WordThatCrossed[start_ind:end_ind,8] = WordThatCrossed # word string indicating which word crossed the threshold
        else:
            WordInds_LexSemErr_ThresholdCrossingIteration_ONsize_Frequency_NumSemFeats_ThresholdWasCrossed_CorrectItemCrossed_WordThatCrossed[start_ind:end_ind,0] = np.nan
            orth_overlap = np.dot(wordlist_to_orth(simulation[subsim]['sim_input_bottomup']).T, lexicon.orthmatrix) # find the raw orthographic overlap
            WordInds_LexSemErr_ThresholdCrossingIteration_ONsize_Frequency_NumSemFeats_ThresholdWasCrossed_CorrectItemCrossed_WordThatCrossed[start_ind:end_ind,3] = np.sum(orth_overlap == 3, axis = 1) # retrieve orthographic neighbors relative to the model's lexicon
            WordInds_LexSemErr_ThresholdCrossingIteration_ONsize_Frequency_NumSemFeats_ThresholdWasCrossed_CorrectItemCrossed_WordThatCrossed[start_ind:end_ind,4:6] = np.nan
        
    cond_names = np.vstack([np.tile(conditions_dict['factors'][i], (num_trials_per_condition, 1)) for i in range(number_of_conditions)])
    cond_codes = np.vstack([np.tile(conditions_dict['coding'][i],(num_trials_per_condition,1)) for i in range(number_of_conditions)])


    df1 = pd.DataFrame(WordInds_LexSemErr_ThresholdCrossingIteration_ONsize_Frequency_NumSemFeats_ThresholdWasCrossed_CorrectItemCrossed_WordThatCrossed, columns = 'WordInds_LexSemErr_ThresholdCrossingIteration_ONsize_Frequency_NumSemFeats_ThresholdWasCrossed_CorrectItemCrossed_WordThatCrossed'.split('_'))
    df2 = pd.DataFrame(cond_names, columns = [i+'_name' for i in conditions_dict['col_names']])
    df3 = pd.DataFrame(cond_codes, columns = [i+'_code' for i in conditions_dict['col_names']])

    final_df = pd.concat([df1, df2,df3], axis=1)
    final_df['WordInput'] = sim_input_all
    final_df.to_csv(f'./simulation_csv_files/{simulation_name}_N400_{time_window[0]}_to_{time_window[-1]}_IterationsToThreshold{THRESHOLD}.csv', index = False)
    # return final_df


##### STANDARD SIMULATION #####
# add dummy condition so that it can be passed into `create_simulation_df`
standard_simulation_withdummycondition = {}
standard_simulation_withdummycondition['dummy'] = standard_simulation
std_sim_conditions = {}
std_sim_conditions['col_names'] = np.array(['Dummy'])
std_sim_conditions['factors'] = np.array([['dummy']])
std_sim_conditions['coding'] =  np.array([[0]])
create_simulation_df(standard_simulation_withdummycondition,std_sim_conditions,'Standard_Simulation')


##### REPETITION PRIMING #####

rep_priming_conditions = {}
rep_priming_conditions['col_names'] = np.array(['Repeated'])
rep_priming_conditions['factors'] = np.array([[i] for i in list(rep_priming.keys())])
rep_priming_conditions['coding'] =  np.array([[-0.5],[0.5]])
create_simulation_df(rep_priming,rep_priming_conditions,'RepetitionPriming_Simulation')


##### SEMANTIC PRIMING #####

sem_priming_conditions = {}
sem_priming_conditions['col_names'] = np.array(['SemanticRelatedness'])
sem_priming_conditions['factors'] = np.array([[i] for i in list(sem_priming.keys())])
sem_priming_conditions['coding'] =  np.array([[-0.5],[0.5]])
create_simulation_df(sem_priming,sem_priming_conditions,'SemanticPriming_Simulation')

##### LEXICAL PREDICTABILITY #####

cloze_conditions = {}
cloze_conditions['col_names'] = np.array(['Cloze'])
cloze_conditions['factors'] = np.array([[i] for i in list(cloze_simulations_bottomup.keys())])
cloze_conditions['coding'] =  np.array([[1/1579],[0.25],[0.5],[0.90]])
create_simulation_df(cloze_simulations_bottomup,cloze_conditions,'ClozeProbability_Simulation')


##### LEXICAL PREDICTION VIOLATION #####

lexical_violation_conditions = {}
lexical_violation_conditions['col_names'] = np.array(['Constraint', 'IsExpected'])
lexical_violation_conditions['factors'] = np.array([['LowConstraint','Unexpected'],['HighConstraint','Unexpected'], ['HighConstraint','Expected']])
lexical_violation_conditions['coding'] =  np.array([[-0.5,-0.5],[0.5,-0.5],[0.5,0.5]])
create_simulation_df(lexical_violation,lexical_violation_conditions,'LexicalViolation_Simulation')


##### ANTICIPATORY SEMANTIC OVERLAP #####
semantic_prediction_overlap_copy = semantic_prediction_overlap.copy()
del semantic_prediction_overlap_copy['high_constraint_expected']

semantic_prediction_overlap_conditions = {}
semantic_prediction_overlap_conditions['col_names'] = np.array(['Relatedness'])
semantic_prediction_overlap_conditions['factors'] = np.array([['SemUnrelated'],['SemRelated']])
semantic_prediction_overlap_conditions['coding'] =  np.array([[-0.5],[0.5]])
create_simulation_df(semantic_prediction_overlap_copy,semantic_prediction_overlap_conditions,'SemanticPredictionOverlap_Simulation')

##### COMPETITION SIMULATION #####
# comp_sims
semicon_conditions = {}
semicon_conditions['col_names'] = np.array(['SemRelatedtoCompetitor', 'HasCompetitor', 'Constraint'])
semicon_conditions['factors'] = np.array([['Unrelated','NoCompetitor', 'Low'],['Unrelated','HasCompetitor', 'High'],['Related','HasCompetitor', 'High'], ['Unrelated','NoCompetitor', 'High']])
semicon_conditions['coding'] =  np.array([[-0.5,-0.5,-0.5],[-0.5,0.5,0.5],[0.5,0.5,0.5], [-0.5,-0.5,0.5]])
create_simulation_df(comp_sims,semicon_conditions,'Competition_Simulation')

############



########### Timing of N400 compared to behavioral threshold crossing


##### define thresholds

standard_simulation_full = run_simulation(sim_input_bottomup = standard_stims[:num_stims], clamp_iterations = NUM_ITERS, sim_filename = 'standard_simulation_full_090826', get_competitor_values = True)
d = standard_simulation_full['simulation_data']

target_lex = d['extract_target_lex_state'][0]
competitor_lex = d['extract_lex_state_mostactive_competitor_state'][0]
target_sem = d['extract_target_sem_state'][0]
nontarget_sem = d['extract_nontarget_sem_state'][0]

n_iters = target_lex.shape[1]

def first_crossing(activity, threshold):
    crossed = activity > threshold
    return np.where(
        crossed.any(axis=1),
        crossed.argmax(axis=1),
        n_iters
    )


# 1. Early lexical recognition:
# lowest threshold where target crosses before competitor on >50% of trials
lex_grid = np.arange(0.01, 0.10, 0.01)

for t1 in lex_grid:
    target_cross = first_crossing(target_lex, t1)
    competitor_cross = first_crossing(competitor_lex, t1)

    if np.mean(target_cross < competitor_cross) > 0.5:
        break


# 2. Initial semantic access:
# lexical threshold whose crossing best matches the largest semantic increase
sem_grid = np.arange(0.1, 4.1, 0.1)

total_sem = target_sem + nontarget_sem
semantic_access_iter = np.argmax(np.diff(total_sem, axis=1), axis=1) + 1

best_mse = np.inf

for threshold in sem_grid:
    target_cross = first_crossing(target_lex, threshold)
    valid = target_cross < n_iters

    if valid.any():
        mse = np.mean(
            (target_cross[valid] - semantic_access_iter[valid]) ** 2
        )

        if mse < best_mse:
            best_mse = mse
            t2 = threshold


# 3. Lexico-semantic selection:
# lowest threshold where target semantics are >=60% at crossing
for t3 in sem_grid:
    target_cross = first_crossing(target_lex, t3)
    valid = target_cross < n_iters

    if not valid.any():
        continue

    rows = np.arange(len(target_cross))[valid]
    cols = target_cross[valid]

    target_prop = (
        target_sem[rows, cols] /
        total_sem[rows, cols]
    )

    if np.all(target_prop >= 0.60):
        break

print(f"lex_identification_threshold = {t1:.2f}, semantic_access_threshold = {t2:.2f}, lexicosemantic_selection_threshold = {t3:.2f}")





# ---- thresholds and labels ----

standard_simulation_full['simulation_data']['total_lexsem_err'] = standard_simulation_full['simulation_data']['total_lex_err'] + standard_simulation_full['simulation_data']['total_sem_err']
pe = standard_simulation_full['simulation_data']['total_lexsem_err'][0]
N400_latency = np.argmax(pe, axis=-1)  # (num_trials,)

standard_simulation_full['simulation_data']['mostactive_lex'] = np.maximum(standard_simulation_full['simulation_data']['extract_target_lex_state'][0],
                                                                      standard_simulation_full['simulation_data']['extract_lex_state_mostactive_competitor_state'][0])

lexst = standard_simulation_full['simulation_data']['mostactive_lex']
t1 = 0.08
t2 = 0.6
t3 = 2.8

iters = np.arange(21)
t1_iters = np.argmax(lexst > t1, axis = -1)
t2_iters = np.argmax(lexst > t2, axis = -1)
t3_iters = np.argmax(lexst > t3, axis = -1)

t1_diff = t1_iters[t1_iters != 0] - N400_latency[t1_iters != 0]
t2_diff = t2_iters[t2_iters != 0] - N400_latency[t2_iters != 0]
t3_diff = t3_iters[t3_iters != 0] - N400_latency[t3_iters != 0]

from scipy.stats import ttest_1samp

for name, thresh in zip(
    ['Early', 'Intermediate', 'Late'],
    [t1, t2, t3]
):
    crossing = np.argmax(lexst > thresh, axis=-1)

    # Exclude trials where threshold was never crossed
    mask = crossing != 0

    # Negative = before N400 peak
    # Positive = after N400 peak
    diff = crossing[mask] - N400_latency[mask]

    t, p = ttest_1samp(diff, 0)

    print(
        f"{name}: "
        f"median = {np.median(diff):.0f} iterations, "
        f"t = {t:.2f}, "
        f"p = {p:.4g}, "
        f"n = {len(diff)}"
    )



fig, (ax0,ax1) = plt.subplots(2,1, figsize = (11, 9))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
ax0.plot(pe.mean(0)/np.max(pe.mean(0))*np.max(lexst.mean(0)),color = 'k',linestyle=':', label = 'Lexico-semantic PE (Average)', linewidth=2.0)
ax0.plot(lexst.mean(0),color = 'gray', label = 'Most Active Lexical State (Average)', linewidth=2.0)
ax0.set_xlabel('Iteration')
ax0.set_ylabel('Activity (scaled)')
ax0.set_xticks(np.arange(lexst.shape[-1]))


ax0.legend(fontsize=11, frameon=False)
ax0.spines['right'].set_visible(False)
ax0.spines['top'].set_visible(False)
for thresh, color in zip([t1,t2,t3], colors):
    crossing = np.argmax(lexst.mean(0) > thresh)
    ax0.hlines(thresh, xmin=0, xmax=crossing-.5, color=color, linewidth=2.0, linestyle='--', alpha=0.8)
labels = ['Low Threshold (early lexical recognition)', 'Intermediate Threshold (initial semantic access)', 'High Threshold (lexico-semantic selection)']

for i, (d, label, color) in enumerate(
    zip([t1_diff, t2_diff, t3_diff], labels, colors)
):
    d_clean = d[~np.isnan(d)]

    parts = ax1.violinplot(
        d_clean,
        positions=[i],
        vert=False,
        showmedians=True
    )

    for body in parts['bodies']:
        body.set_facecolor(color)
        body.set_alpha(0.5)

    ax1.text(
        ax1.get_xlim()[1] + 0.3,
        i,
        label,
        va='center',
        fontsize=11
    )

ax1.axvline(0, color='k', linewidth=0.8)
ax1.set_yticks([])
ax1.set_xlabel('Threshold Crossing Iteration minus N400 Latency')
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)

ax0.text(
    -0.12, 1.05, '(a)',
    transform=ax0.transAxes,
    fontsize=14,
    fontweight='bold',
    va='top',
    ha='left'
)

ax1.text(
    -0.12, 1.05, '(b)',
    transform=ax1.transAxes,
    fontsize=14,
    fontweight='bold',
    va='top',
    ha='left'
)
# Same x-axis scale: 20 iteration units across each panel
n_iter = lexst.shape[-1] - 1   # e.g. 20

ax0.set_xlim(0, n_iter)
ax0.set_xticks(np.arange(0, n_iter + 1, 1))

ax1.set_xlim(-5, -5 + n_iter)
ax1.set_xticks(np.arange(-5, -5 + n_iter + 1, 1))

fig.suptitle(
    'Relative timing of N400 peak and behavioral responses'
)
fig.tight_layout()

fig.savefig("./plots/threshold_plots/Figure12_n400_vs_threshold_timing.png", dpi=300, bbox_inches="tight")
fig.savefig("./plots/threshold_plots/Figure12_n400_vs_threshold_timing.svg", dpi=300, bbox_inches="tight")
plt.close()



###############################################################################
# REFERENCE-STYLE FIGURES
###############################################################################

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


###############################################################################
# GLOBAL PLOTTING SETTINGS
###############################################################################

PLOT_THRESHOLD = 2.8

PLOT_DIR = "./plots/results"
os.makedirs(PLOT_DIR, exist_ok=True)


###############################################################################
# LINE STYLES
###############################################################################

BLACK_SOLID = dict(
    color="k",
    linestyle="-"
)

BLACK_DASHED = dict(
    color="k",
    linestyle="--"
)

BLACK_DOTTED = dict(
    color="k",
    linestyle=":"
)

BLACK_DASHDOT = dict(
    color="k",
    linestyle="-."
)


# Custom red dashed line:
# linestyle = (dash phase, (dash length, gap length))
RED_CUSTOM_DASHED = dict(
    color="r",
    linestyle=(4, (8, 4))
)


# Competition styles
BLUE_DOTTED = dict(
    color="b",
    linestyle=":"
)

ORANGE_CUSTOM_DASHED = dict(
    color="#ffa500",
    linestyle=(0, (8, 4))
)

###############################################################################
# MAIN PLOTTING FUNCTION
###############################################################################
def plot_results(
    filename,
    simulation,
    conditions,
    labels,
    styles,
    title=None,
    threshold=2.8,
    num_target_iters=20,
    inset_prestim_iters=4,
    main_prestim_iters=None,
    reference_pre_iters=20,
    plot_dir="./plots/results"
):

    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    os.makedirs(plot_dir, exist_ok=True)

    if not (
        len(conditions)
        == len(labels)
        == len(styles)
    ):
        raise ValueError(
            "conditions, labels, and styles must have equal length."
        )

    def get_trial_data(condition, variable):

        data = np.asarray(
            simulation[condition]["simulation_data"][variable]
        )

        if data.ndim == 3:
            data = data[0]

        if data.ndim != 2:
            raise ValueError(
                f"{condition}/{variable}: expected trials x time, got {data.shape}"
            )

        return data

    def get_mean(condition, variable):

        return np.nanmean(
            get_trial_data(condition, variable),
            axis=0
        )

    def get_target_onset(condition, variable):

        data = get_trial_data(
            condition,
            variable
        )

        n_timepoints = data.shape[-1]

        target_samples = num_target_iters + 1

        if n_timepoints < target_samples:
            raise ValueError(
                f"{condition}/{variable}: "
                f"{n_timepoints} samples, but target requires "
                f"{target_samples}."
            )

        return n_timepoints - target_samples

    def get_timecourse(condition, variable):

        y = get_mean(
            condition,
            variable
        )

        onset = get_target_onset(
            condition,
            variable
        )

        x = np.arange(len(y)) - onset

        return x, y

    def clean_axis(ax):

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.tick_params(
            direction="out",
            labelsize=13
        )
    # -------------------------------------------------------------------------
    # Infer pre-target duration
    # -------------------------------------------------------------------------

    pre_lengths = []

    for condition in conditions:

        if condition not in simulation:
            raise KeyError(
                f"Condition '{condition}' not found."
            )

        onset = get_target_onset(
            condition,
            "total_lexsem_err"
        )

        pre_lengths.append(onset)

    if len(set(pre_lengths)) != 1:
        raise ValueError(
            f"Conditions in {filename} have different pre-target durations: "
            f"{dict(zip(conditions, pre_lengths))}"
        )

    pre_duration = pre_lengths[0]

    if main_prestim_iters is None:
        plot_pre_duration = pre_duration
    else:
        plot_pre_duration = min(
            main_prestim_iters,
            pre_duration
        )

    print(
        f"{filename}: actual pre-target duration = {pre_duration}; "
        f"displaying = {plot_pre_duration}"
    )

    # -------------------------------------------------------------------------
    # Figure width
    #
    # Full preactivation figures remain wide.
    # Priming figures get a narrower overall canvas.
    # -------------------------------------------------------------------------

    reference_range = (
        reference_pre_iters
        + num_target_iters
    )

    current_range = (
        plot_pre_duration
        + num_target_iters
    )

    range_ratio = current_range / reference_range

    if main_prestim_iters is None:
        fig_width = 12.5
    else:
        fig_width = 9.5

    fig = plt.figure(
        figsize=(fig_width, 8)
    )

    # -------------------------------------------------------------------------
    # Layout
    # -------------------------------------------------------------------------

    if main_prestim_iters is None:

        # Wide preactivation layout
        main_left = 0.08
        main_width = 0.58

        inset_left = 0.75
        inset_width = 0.20

        legend_x = 0.82

    else:

        # More compact repetition / semantic priming layout
        main_left = 0.09
        main_width = 0.56

        inset_left = 0.72
        inset_width = 0.24

        legend_x = 0.82

    ax_total = fig.add_axes([
        main_left,
        0.57,
        main_width,
        0.36
    ])

    ax_sem = fig.add_axes([
        inset_left,
        0.75,
        inset_width,
        0.17
    ])

    ax_lex = fig.add_axes([
        inset_left,
        0.51,
        inset_width,
        0.17
    ])

    ax_state = fig.add_axes([
        main_left,
        0.09,
        main_width,
        0.35
    ])

    legend_handles = []

    # -------------------------------------------------------------------------
    # Plot conditions
    # -------------------------------------------------------------------------

    for condition, label, style in zip(
        conditions,
        labels,
        styles
    ):

        # Total lexico-semantic PE
        x_total, y_total = get_timecourse(
            condition,
            "total_lexsem_err"
        )

        ax_total.plot(
            x_total,
            y_total,
            linewidth=2.5,
            **style
        )

        legend_handles.append(
            Line2D(
                [0],
                [0],
                linewidth=2.5,
                label=label,
                **style
            )
        )

        # Lexical PE inset
        x_lex, y_lex = get_timecourse(
            condition,
            "total_lex_err"
        )

        keep_lex = (
            (x_lex >= -inset_prestim_iters)
            &
            (x_lex <= num_target_iters)
        )

        ax_lex.plot(
            x_lex[keep_lex],
            y_lex[keep_lex],
            linewidth=2.0,
            **style
        )

        # Semantic PE inset
        x_sem, y_sem = get_timecourse(
            condition,
            "total_sem_err"
        )

        keep_sem = (
            (x_sem >= -inset_prestim_iters)
            &
            (x_sem <= num_target_iters)
        )

        ax_sem.plot(
            x_sem[keep_sem],
            y_sem[keep_sem],
            linewidth=2.0,
            **style
        )

        # Most active lexical state
        x_state, y_state = get_timecourse(
            condition,
            "max_lex_state_activation"
        )

        ax_state.plot(
            x_state,
            y_state,
            linewidth=2.5,
            **style
        )

    # -------------------------------------------------------------------------
    # Main Total PE
    # -------------------------------------------------------------------------

    clean_axis(ax_total)

    ax_total.set_xlim(
        -plot_pre_duration,
        num_target_iters
    )

    ax_total.set_xlabel("Iterations", fontsize = 14)
    ax_total.set_ylabel("Total\nLexico-semantic PE", fontsize = 14)

    ax_total.axvline(
        0,
        color="0.85",
        linewidth=0.8,
        zorder=0
    )

    ax_total.text(
        -0.14,
        1.04,
        "(a)",
        transform=ax_total.transAxes,
        fontsize=18,
        fontweight="bold"
    )

    # -------------------------------------------------------------------------
    # Lexical inset
    # -------------------------------------------------------------------------

    clean_axis(ax_lex)

    ax_lex.set_title(
        "Lexical PE",
        fontsize=14
    )

    ax_lex.set_xlim(
        -inset_prestim_iters,
        num_target_iters
    )

    ax_lex.set_xticks([
        -4,
        0,
        5,
        10,
        15,
        20
    ])

    ax_lex.axvline(
        0,
        color="0.80",
        linewidth=0.8,
        zorder=0
    )

    # -------------------------------------------------------------------------
    # Semantic inset
    # -------------------------------------------------------------------------

    clean_axis(ax_sem)

    ax_sem.set_title(
        "Semantic PE",
        fontsize=14
    )

    ax_sem.set_xlim(
        -inset_prestim_iters,
        num_target_iters
    )

    ax_sem.set_xticks([
        -4,
        0,
        5,
        10,
        15,
        20
    ])

    ax_lex.set_xlabel("Iterations")

    ax_sem.axvline(
        0,
        color="0.80",
        linewidth=0.8,
        zorder=0
    )

    # -------------------------------------------------------------------------
    # Lexical state
    # -------------------------------------------------------------------------

    clean_axis(ax_state)

    ax_state.set_xlim(
        -plot_pre_duration,
        num_target_iters
    )

    ax_state.set_xlabel("Iterations", fontsize = 14)
    ax_state.set_ylabel("Most Active Lexical State", fontsize = 14)

    ax_state.axvline(
        0,
        color="0.85",
        linewidth=0.8,
        zorder=0
    )

    ax_state.axhline(
        threshold,
        color="0.60",
        linestyle=":",
        linewidth=1.5,
        zorder=0
    )

    ax_state.text(
        -plot_pre_duration + 0.2,
        threshold + 0.08,
        "response\nthreshold",
        color="0.50",
        fontsize=10
    )

    ax_state.text(
        -0.14,
        1.04,
        "(b)",
        transform=ax_state.transAxes,
        fontsize=18,
        fontweight="bold"
    )

    # -------------------------------------------------------------------------
    # Legend
    # -------------------------------------------------------------------------

    fig.legend(
        handles=legend_handles,
        frameon=False,
        fontsize=13,
        loc="center",
        bbox_to_anchor=(
            legend_x,
            0.29
        ),
        handlelength=3
    )

    # -------------------------------------------------------------------------
    # Title
    # -------------------------------------------------------------------------

    if title is not None:

        fig.suptitle(
            title,
            fontsize=20,
            y=0.99
        )

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------

    png_file = os.path.join(
        plot_dir,
        filename + ".png"
    )

    svg_file = os.path.join(
        plot_dir,
        filename + ".svg"
    )

    fig.savefig(
        png_file,
        dpi=300,
        bbox_inches="tight"
    )

    fig.savefig(
        svg_file,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(f"Finished: {filename}")
###############################################################################
# 1. REPETITION PRIMING
#
# Main panels: -5 to 20
# Insets:      -4 to 20
# Main panels are physically narrower.
###############################################################################

plot_results(
    filename="Figure6_repetition_priming",

    simulation=rep_priming,

    conditions=[
        "unrepeated",
        "repeated"
    ],

    labels=[
        "Non-repeated",
        "Repeated"
    ],

    styles=[
        BLACK_DASHED,
        BLACK_SOLID
    ],

    title="Repetition Priming",

    main_prestim_iters=4
)


###############################################################################
# 2. SEMANTIC PRIMING
#
# Main panels: -5 to 20
# Insets:      -4 to 20
# Main panels are physically narrower.
###############################################################################

plot_results(
    filename="Figure7_semantic_priming",

    simulation=sem_priming,

    conditions=[
        "semunrelated",
        "semrelated"
    ],

    labels=[
        "Unrelated",
        "Semantically Related"
    ],

    styles=[
        BLACK_DASHED,
        BLACK_SOLID
    ],

    title="Semantic Priming",

    main_prestim_iters=4
)


###############################################################################
# 3. LEXICAL PROBABILITY
###############################################################################

plot_results(
    filename="Figure8_lexical_probability",

    simulation=cloze_simulations_bottomup,

    conditions=[
        "low_cloze",
        "med_low_cloze",
        "med_high_cloze",
        "high_cloze"
    ],

    labels=[
        "0.06%",
        "25%",
        "50%",
        "90%"
    ],

    styles=[
        BLACK_DOTTED,
        BLACK_DASHDOT,
        BLACK_DASHED,
        BLACK_SOLID
    ],

    title="Lexical Probability"
)


###############################################################################
# 4. LEXICAL PREDICTION VIOLATION
###############################################################################

plot_results(
    filename="lexical_prediction_violation",

    simulation=lexical_violation,

    conditions=[
        "low_constraint_unexpected",
        "high_constraint_unexpected",
        "high_constraint_expected"
    ],

    labels=[
        "Low Constraint Unexpected",
        "High Constraint Unexpected",
        "High Constraint Expected"
    ],

    styles=[
        BLACK_DOTTED,
        RED_CUSTOM_DASHED,
        BLACK_SOLID
    ],

    title="Lexical Prediction Violation"
)


###############################################################################
# 5. ANTICIPATORY SEMANTIC OVERLAP
###############################################################################

plot_results(
    filename="semantic_prediction_overlap",

    simulation=semantic_prediction_overlap,

    conditions=[
        "semunrelated_90cloze",
        "semrelated_90cloze",
        "high_constraint_expected"
    ],

    labels=[
        "90% Constraint Unrelated",
        "90% Constraint Related",
        "Expected"
    ],

    styles=[
        BLACK_DOTTED,
        BLACK_DASHED,
        BLACK_SOLID
    ],

    title="Anticipatory Semantic Overlap"
)


###############################################################################
# 6. MORE PROBABLE COMPETITOR
###############################################################################
def plot_results(
    filename,
    simulation,
    conditions,
    labels,
    styles,
    title=None,
    threshold=2.8,
    num_target_iters=20,
    inset_prestim_iters=4,
    main_prestim_iters=None,
    reference_pre_iters=20,
    plot_dir="./plots/results"
):

    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    os.makedirs(plot_dir, exist_ok=True)

    if not (
        len(conditions)
        == len(labels)
        == len(styles)
    ):
        raise ValueError(
            "conditions, labels, and styles must have equal length."
        )

    def get_trial_data(condition, variable):

        data = np.asarray(
            simulation[condition]["simulation_data"][variable]
        )

        if data.ndim == 3:
            data = data[0]

        if data.ndim != 2:
            raise ValueError(
                f"{condition}/{variable}: expected trials x time, got {data.shape}"
            )

        return data

    def get_mean(condition, variable):

        return np.nanmean(
            get_trial_data(condition, variable),
            axis=0
        )

    def get_target_onset(condition, variable):

        data = get_trial_data(
            condition,
            variable
        )

        n_timepoints = data.shape[-1]

        target_samples = num_target_iters + 1

        if n_timepoints < target_samples:
            raise ValueError(
                f"{condition}/{variable}: "
                f"{n_timepoints} samples, but target requires "
                f"{target_samples}."
            )

        return n_timepoints - target_samples

    def get_timecourse(condition, variable):

        y = get_mean(
            condition,
            variable
        )

        onset = get_target_onset(
            condition,
            variable
        )

        x = np.arange(len(y)) - onset

        return x, y

    def clean_axis(ax):

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(direction="out", labelsize = 13)

    # -------------------------------------------------------------------------
    # Infer pre-target duration
    # -------------------------------------------------------------------------

    pre_lengths = []

    for condition in conditions:

        if condition not in simulation:
            raise KeyError(
                f"Condition '{condition}' not found."
            )

        onset = get_target_onset(
            condition,
            "total_lexsem_err"
        )

        pre_lengths.append(onset)

    if len(set(pre_lengths)) != 1:
        raise ValueError(
            f"Conditions in {filename} have different pre-target durations: "
            f"{dict(zip(conditions, pre_lengths))}"
        )

    pre_duration = pre_lengths[0]

    if main_prestim_iters is None:
        plot_pre_duration = pre_duration
    else:
        plot_pre_duration = min(
            main_prestim_iters,
            pre_duration
        )

    print(
        f"{filename}: actual pre-target duration = {pre_duration}; "
        f"displaying = {plot_pre_duration}"
    )

    # -------------------------------------------------------------------------
    # Figure width
    #
    # Full preactivation figures remain wide.
    # Priming figures get a narrower overall canvas.
    # -------------------------------------------------------------------------

    reference_range = (
        reference_pre_iters
        + num_target_iters
    )

    current_range = (
        plot_pre_duration
        + num_target_iters
    )

    range_ratio = current_range / reference_range

    if main_prestim_iters is None:
        fig_width = 12.5
    else:
        fig_width = 9.5

    fig = plt.figure(
        figsize=(fig_width, 8)
    )

    # -------------------------------------------------------------------------
    # Layout
    # -------------------------------------------------------------------------

    if main_prestim_iters is None:

        # Wide preactivation layout
        main_left = 0.08
        main_width = 0.58

        inset_left = 0.75
        inset_width = 0.20

        legend_x = 0.82

    else:

        # More compact repetition / semantic priming layout
        main_left = 0.09
        main_width = 0.56

        inset_left = 0.72
        inset_width = 0.24

        legend_x = 0.82

    ax_total = fig.add_axes([
        main_left,
        0.57,
        main_width,
        0.36
    ])

    ax_sem = fig.add_axes([
        inset_left,
        0.75,
        inset_width,
        0.17
    ])

    ax_lex = fig.add_axes([
        inset_left,
        0.51,
        inset_width,
        0.17
    ])

    ax_state = fig.add_axes([
        main_left,
        0.09,
        main_width,
        0.35
    ])

    legend_handles = []

    # -------------------------------------------------------------------------
    # Plot conditions
    # -------------------------------------------------------------------------

    for condition, label, style in zip(
        conditions,
        labels,
        styles
    ):

        # Total lexico-semantic PE
        x_total, y_total = get_timecourse(
            condition,
            "total_lexsem_err"
        )

        ax_total.plot(
            x_total,
            y_total,
            linewidth=2.5,
            **style
        )

        legend_handles.append(
            Line2D(
                [0],
                [0],
                linewidth=2.5,
                label=label,
                **style
            )
        )

        # Lexical PE inset
        x_lex, y_lex = get_timecourse(
            condition,
            "total_lex_err"
        )

        keep_lex = (
            (x_lex >= -inset_prestim_iters)
            &
            (x_lex <= num_target_iters)
        )

        ax_lex.plot(
            x_lex[keep_lex],
            y_lex[keep_lex],
            linewidth=2.0,
            **style
        )

        # Semantic PE inset
        x_sem, y_sem = get_timecourse(
            condition,
            "total_sem_err"
        )

        keep_sem = (
            (x_sem >= -inset_prestim_iters)
            &
            (x_sem <= num_target_iters)
        )

        ax_sem.plot(
            x_sem[keep_sem],
            y_sem[keep_sem],
            linewidth=2.0,
            **style
        )

        # Most active lexical state
        x_state, y_state = get_timecourse(
            condition,
            "max_lex_state_activation"
        )

        ax_state.plot(
            x_state,
            y_state,
            linewidth=2.5,
            **style
        )

    # -------------------------------------------------------------------------
    # Main Total PE
    # -------------------------------------------------------------------------

    clean_axis(ax_total)

    ax_total.set_xlim(
        -plot_pre_duration,
        num_target_iters
    )

    ax_total.set_xlabel("Iterations")
    ax_total.set_ylabel("Total\nLexico-semantic PE")

    ax_total.axvline(
        0,
        color="0.85",
        linewidth=0.8,
        zorder=0
    )

    ax_total.text(
        -0.14,
        1.04,
        "(a)",
        transform=ax_total.transAxes,
        fontsize=18,
        fontweight="bold"
    )

    # -------------------------------------------------------------------------
    # Lexical inset
    # -------------------------------------------------------------------------

    clean_axis(ax_lex)

    ax_lex.set_title(
        "Lexical PE",
        fontsize=14
    )

    ax_lex.set_xlim(
        -inset_prestim_iters,
        num_target_iters
    )

    ax_lex.set_xticks([
        -4,
        0,
        5,
        10,
        15,
        20
    ])

    ax_lex.axvline(
        0,
        color="0.80",
        linewidth=0.8,
        zorder=0
    )

    # -------------------------------------------------------------------------
    # Semantic inset
    # -------------------------------------------------------------------------

    clean_axis(ax_sem)

    ax_sem.set_title(
        "Semantic PE",
        fontsize=14
    )

    ax_sem.set_xlim(
        -inset_prestim_iters,
        num_target_iters
    )

    ax_sem.set_xticks([
        -4,
        0,
        5,
        10,
        15,
        20
    ])

    ax_lex.set_xlabel("Iterations")

    ax_sem.axvline(
        0,
        color="0.80",
        linewidth=0.8,
        zorder=0
    )

    # -------------------------------------------------------------------------
    # Lexical state
    # -------------------------------------------------------------------------

    clean_axis(ax_state)

    ax_state.set_xlim(
        -plot_pre_duration,
        num_target_iters
    )

    ax_state.set_xlabel("Iterations", fontsize = 14)
    ax_state.set_ylabel("Most Active Lexical State", fontsize = 14)

    ax_state.axvline(
        0,
        color="0.85",
        linewidth=0.8,
        zorder=0
    )

    ax_state.axhline(
        threshold,
        color="0.60",
        linestyle=":",
        linewidth=1.5,
        zorder=0
    )

    ax_state.text(
        -plot_pre_duration + 0.2,
        threshold + 0.08,
        "response threshold",
        color="0.50",
        fontsize=10
    )

    ax_state.text(
        -0.14,
        1.04,
        "(b)",
        transform=ax_state.transAxes,
        fontsize=18,
        fontweight="bold"
    )

    # -------------------------------------------------------------------------
    # Legend
    # -------------------------------------------------------------------------

    fig.legend(
        handles=legend_handles,
        frameon=False,
        fontsize=13,
        loc="center",
        bbox_to_anchor=(
            legend_x,
            0.29
        ),
        handlelength=3
    )

    # -------------------------------------------------------------------------
    # Title
    # -------------------------------------------------------------------------

    if title is not None:

        fig.suptitle(
            title,
            fontsize=17,
            y=0.99
        )

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------

    png_file = os.path.join(
        plot_dir,
        filename + ".png"
    )

    svg_file = os.path.join(
        plot_dir,
        filename + ".svg"
    )

    fig.savefig(
        png_file,
        dpi=300,
        bbox_inches="tight"
    )

    fig.savefig(
        svg_file,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(f"Finished: {filename}")

###############################################################################
# 6. MORE PROBABLE COMPETITOR
###############################################################################

plot_results(
    filename="Figure10_competition",

    simulation=comp_sims,

    conditions=[
        "no_comp",
        "unrel_comp",
        "rel_comp",
        "expected"
    ],

    labels=[
'Less Expected (30%) without Competitor',
     'Less Expected (30%) with Unrelated Competitor (70%)',
     'Less Expected (30%) with Related Competitor (70%)',
     'More Expected (70%)'
    ],

    styles=[
        BLUE_DOTTED,
        ORANGE_CUSTOM_DASHED,
        BLACK_DASHED,
        BLACK_SOLID
    ],

    title="Effects of a More Probable Competitor"
)
###############################################################################
# 7. COMBINED PREDICTION-VIOLATION FIGURE
###############################################################################

merged_prediction_simulation = {

    "low_constraint_unexpected":
        lexical_violation[
            "low_constraint_unexpected"
        ],

    "high_constraint_unexpected":
        lexical_violation[
            "high_constraint_unexpected"
        ],

    "semrelated_90cloze":
        semantic_prediction_overlap[
            "semrelated_90cloze"
        ],

    "high_constraint_expected":
        lexical_violation[
            "high_constraint_expected"
        ]
}


plot_results(
    filename="Figure9_prediction_violation_combined",

    simulation=merged_prediction_simulation,

    conditions=[
        "low_constraint_unexpected",
        "high_constraint_unexpected",
        "semrelated_90cloze",
        "high_constraint_expected"
    ],

    labels=[
        "Low Constraint Unexp. Unrel.",
        "High Constraint Unexp. Unrel.",
        "High Constraint Unexp. Related",
        "High Constraint Expected"
    ],

    styles=[
        BLACK_DOTTED,
        RED_CUSTOM_DASHED,
        BLACK_DASHED,
        BLACK_SOLID
    ],

    title="No Cost for Unrelated Words, Benefit for Related Words"
)




standard_simulation_full = run_simulation(sim_input_bottomup = standard_stims[:num_stims], clamp_iterations = NUM_ITERS, sim_filename = 'standard_simulation_full_090826', get_competitor_values = True)

data = standard_simulation_full['simulation_data']

padder = lambda arr,numzeros: np.insert(arr, 0,np.repeat(0,numzeros)) 
padder = lambda arr,numzeros: np.insert(arr, 0,np.repeat(np.min(arr),numzeros)) 
padder = lambda arr,numzeros: np.insert(arr, 0,np.repeat(arr[0],numzeros)) 

plotter = lambda ax, idx, arr, color, linestyle: ax[idx].plot(np.arange(-2, 21), padder(arr[0,:,:].mean(0), 2), color = color, linestyle = linestyle)

arrs = [
    'extract_target_lex_state',
'extract_lex_state_ON1_competitor_state',
'extract_lex_state_ON2_competitor_state',
'extract_lex_state_ON3_competitor_state',
'extract_lex_state_mostactive_competitor_state',
]
arrs_ctx = [
    'extract_target_ctx_state',
'extract_ctx_state_ON1_competitor_state',
'extract_ctx_state_ON2_competitor_state',
'extract_ctx_state_ON3_competitor_state',
'extract_ctx_state_mostactive_competitor_state',
]
arrs_err = [
    'extract_target_lex_error',
'extract_lex_error_ON1_competitor_error',
'extract_lex_error_ON2_competitor_error',
'extract_lex_error_ON3_competitor_error',
'extract_lex_error_mostactive_competitor_error',
]
cols = [
    'k',
    'k',
    'k',
    'k',
    'r',
]
styles = [
    '-',
    ':',
    '-.',
    '--',
    '-',
]
labels = [
    'Target Lexical Unit',
    '1-letter Neighbor',
    '2-letter Neighbor',
    '3-letter Neighbor',
    'Strongest Lexical Competitor',
]
labels_ctx = [
    'Target Conceptual Unit',
    '1-letter Neighbor',
    '2-letter Neighbor',
    '3-letter Neighbor',
    'Strongest Conceptual Competitor',
]
fig, ax = plt.subplots(5,1, figsize = (10,8))

ax[0].set_ylabel('Conceptual\nState', fontsize=14)

for arry, col, style,label in zip(arrs_ctx,cols,styles, labels_ctx):
    color = 'r' if label == 'Strongest Conceptual Competitor' else  '#710193'
    ax[0].plot(np.arange(-2, 21), padder(np.nanmean(data[arry][0,:,:],axis=0), 2), color = color, linestyle = style, label = label)

plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon = False)

ax[1].plot(np.arange(-2, 21), padder(data['total_sem_err'][0].mean(0),2), color = '#ff00ff', alpha = 0.2, label = 'Total')
ax[1].plot(np.arange(-2,21), 
           padder(np.nanmean(data['extract_target_sem_error'][0,:,:],axis=0),2), color = '#ff00ff', label = 'Target-consistent Semantic Mass')
ax[1].plot(np.arange(-2,21), 
           padder(np.nanmean(data['extract_nontarget_sem_error'][0,:,:],axis=0),2), color = '#ff00ff', linestyle = '--', label = 'Target-inconsistent Semantic Mass')
ax[1].set_ylabel('Semantic\nError', fontsize=14)


ax[2].plot(np.arange(-2,21), 
           padder(np.nanmean(data['extract_target_sem_state'][0,:,:],axis=0),2), color = '#ff00ff', label = 'Target-consistent Semantic Mass')
ax[2].plot(np.arange(-2,21), 
           padder(np.nanmean(data['extract_nontarget_sem_state'][0,:,:],axis=0),2), color = '#ff00ff', linestyle = '--', label = 'Target-inconsistent Semantic Mass')
ax[2].set_ylabel('Semantic\nState', fontsize=14)

ax[3].plot(np.arange(-2, 21), padder(data['total_lex_err'][0].mean(0),2), color = 'k', alpha = 0.2, label = 'Total')

for arry, col, style,label in zip(arrs_err,cols,styles, labels):
    ax[3].plot(np.arange(-2, 21), padder(np.nanmean(data[arry][0,:,:],axis=0), 2), color = col, linestyle = style, label = label)
ax[3].set_ylabel('Lexical\nError', fontsize=14)
for arry, col, style,label in zip(arrs,cols,styles, labels):
    ax[4].plot(np.arange(-2, 21), padder(np.nanmean(data[arry][0,:,:],axis=0), 2), color = col, linestyle = style, label = label)
[ax[idx].spines[['right', 'top']].set_visible(False) for idx in range(5)]
ax[4].set_xlabel('Iteration number (relative to onset)')
ax[4].set_ylabel('Lexical\nState', fontsize=14)

plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon = False)

for a in ax:
    a.set_xticks(np.arange(-2, 21))
    a.axvline(0, color = 'k', linestyle = '--')
    a.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)

plt.tight_layout()
plt.savefig('./plots/Figure4_090826.png', dpi = 300)
plt.savefig('./plots/Figure4_090826.svg', dpi = 300)
plt.close()

