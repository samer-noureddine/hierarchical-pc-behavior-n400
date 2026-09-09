import numpy as np
from orth_neighborhood_utils import *

def get_correct_inds(stimset, model_lexicon):
    # given a list of strings, return their indices in the model's lexicon (if they exist; otherwise return nan).
    lexicon_ind_list = []
    for item in stimset:
        if item not in model_lexicon:
            lexicon_ind_list.append(np.nan)
        else:
            # grab everything about this item's orthography, lexical and semantic indices
            lexicon_ind_list.append(list(model_lexicon).index(item))
    return np.array(lexicon_ind_list)

def get_summary(model):
    '''
    Takes in a model at a given iteration and records summary values from the model (e.g. total lexico-semantic PE) at that iteration
    ''' 

    # The first four dimensions of model.statespace[level]: 
    kinds = ['state', 'reconstruction', 'preactivation', 'prediction_error'] 

    summary_dict = {}

    target_representation = {'orth': model.lexicon.orthmatrix.T,
                                'lex': model.lexicon.lexicalmatrix.T,
                                'sem': model.lexicon.semfeatmatrix.T}
    if model.get_competitor_values:
        extract_target_lex_state = np.zeros(model.sim_input_bottomup.shape[0])
        extract_lex_state_mostactive_competitor_state = np.zeros(model.sim_input_bottomup.shape[0])
        extract_lex_state_ON1_competitor_state = np.zeros(model.sim_input_bottomup.shape[0])
        extract_lex_state_ON2_competitor_state = np.zeros(model.sim_input_bottomup.shape[0])
        extract_lex_state_ON3_competitor_state = np.zeros(model.sim_input_bottomup.shape[0])
        extract_lex_err_competitor_state = np.zeros(model.sim_input_bottomup.shape[0])
        
        extract_target_sem_state = np.zeros(model.sim_input_bottomup.shape[0])
        extract_nontarget_sem_state = np.zeros(model.sim_input_bottomup.shape[0])
        extract_sem_state_competitor_state = np.zeros(model.sim_input_bottomup.shape[0])
        extract_sem_err_competitor_state = np.zeros(model.sim_input_bottomup.shape[0])
        extract_ctx_state_competitor_state = np.zeros(model.sim_input_bottomup.shape[0])
        
        extract_ctx_state_mostactive_competitor_state = np.zeros(model.sim_input_bottomup.shape[0])
        extract_target_ctx_state = np.zeros(model.sim_input_bottomup.shape[0])
        extract_nontarget_ctx_state = np.zeros(model.sim_input_bottomup.shape[0])
        extract_ctx_state_ON1_competitor_state = np.zeros(model.sim_input_bottomup.shape[0])
        extract_ctx_state_ON2_competitor_state = np.zeros(model.sim_input_bottomup.shape[0])
        extract_ctx_state_ON3_competitor_state = np.zeros(model.sim_input_bottomup.shape[0])

        orth_overlap = np.load("./helper_txt_files/orth_overlap.npy")  # Load
        sem_overlap = np.load("./helper_txt_files/sem_overlap.npy")  # Load
        get_ON_1 = lambda target_idx : np.argsort(orth_overlap[target_idx,:])[np.sort(orth_overlap[target_idx,:]) == 1][::-1]
        get_ON_2 = lambda target_idx : np.argsort(orth_overlap[target_idx,:])[np.sort(orth_overlap[target_idx,:]) == 2][::-1]
        get_ON_3 = lambda target_idx : np.argsort(orth_overlap[target_idx,:])[np.sort(orth_overlap[target_idx,:]) == 3][::-1]
        get_SN = lambda target_idx : np.argsort(sem_overlap[target_idx,:])[np.sort(sem_overlap[target_idx,:]) > 0][::-1][1:]
        get_SF_inds = lambda target_idx : np.where(model.lexicon.semfeatmatrix[:,target_idx])[0]
        get_nontarget_SF_inds = lambda target_idx : np.delete(np.arange(model.lexicon.semfeatmatrix.shape[0]),get_SF_inds(target_idx))

        target_indices = [list(model.lexicon.words).index(w) for w in model.sim_input_bottomup]
        for input_index, target_index in enumerate(target_indices):
            temp_lex_vector_state = model.statespace['lex'][0,:,input_index].copy() #model.lex_update_bottomup[:,input_index].copy() #
            extract_target_lex_state[input_index] = temp_lex_vector_state[target_index].copy()
            # zero out the target values
            temp_lex_vector_state[target_index] = 0
            # get most active lexical state unit competitor
            extract_lex_state_mostactive_competitor_state[input_index] = np.sort(temp_lex_vector_state)[::-1][0]
            # get ON1/2/3 competitor lex state values
            lex_state_ON1_competitors_state = temp_lex_vector_state[get_ON_1(target_indices[input_index])].sum()
            lex_state_ON2_competitors_state = temp_lex_vector_state[get_ON_2(target_indices[input_index])].sum()
            if temp_lex_vector_state[get_ON_3(target_indices[input_index])].size!=0:
                lex_state_ON3_competitors_state = temp_lex_vector_state[get_ON_3(target_indices[input_index])].sum()
            else:
                lex_state_ON3_competitors_state = np.nan
            extract_lex_state_ON1_competitor_state[input_index] = lex_state_ON1_competitors_state
            extract_lex_state_ON2_competitor_state[input_index] = lex_state_ON2_competitors_state
            extract_lex_state_ON3_competitor_state[input_index] = lex_state_ON3_competitors_state
            temp_sem_vector_state = model.statespace['sem'][0,:,input_index].copy() #model.sem_update_bottomup[:,input_index].copy() 
            extract_target_sem_state[input_index] = temp_sem_vector_state[get_SF_inds(target_index)].copy().sum()
            extract_nontarget_sem_state[input_index] = temp_sem_vector_state[get_nontarget_SF_inds(target_index)].sum()

            temp_ctx_vector_state = model.statespace['ctx'][0,:,input_index].copy()
            extract_target_ctx_state[input_index] = temp_ctx_vector_state[target_index].copy()
            # zero out the target values
            temp_ctx_vector_state[target_index] = 0
            # get most active conceptual state unit competitor
            extract_ctx_state_mostactive_competitor_state[input_index] = np.sort(temp_ctx_vector_state)[::-1][0]
            extract_nontarget_ctx_state[input_index] = temp_ctx_vector_state.sum()


            temp_ctx_vector_state = model.statespace['ctx'][0,:,input_index].copy()
            extract_target_ctx_state[input_index] = temp_ctx_vector_state[target_index].copy()
            # zero out the target values
            temp_ctx_vector_state[target_index] = 0
            # get most active ctxical state unit competitor
            extract_ctx_state_mostactive_competitor_state[input_index] = np.sort(temp_ctx_vector_state)[::-1][0]
            # get ON1/2/3 competitor ctx state values
            ctx_state_ON1_competitors_state = temp_ctx_vector_state[get_ON_1(target_indices[input_index])].sum()
            ctx_state_ON2_competitors_state = temp_ctx_vector_state[get_ON_2(target_indices[input_index])].sum()
            if temp_ctx_vector_state[get_ON_3(target_indices[input_index])].size !=0:
                ctx_state_ON3_competitors_state = temp_ctx_vector_state[get_ON_3(target_indices[input_index])].sum()
            else:
                ctx_state_ON3_competitors_state = np.nan

            extract_ctx_state_ON1_competitor_state[input_index] = ctx_state_ON1_competitors_state
            extract_ctx_state_ON2_competitor_state[input_index] = ctx_state_ON2_competitors_state
            extract_ctx_state_ON3_competitor_state[input_index] = ctx_state_ON3_competitors_state


            ####### SAME THING BUT FOR ERROR ######
        extract_target_lex_error = np.zeros(model.sim_input_bottomup.shape[0])
        extract_lex_error_mostactive_competitor_error = np.zeros(model.sim_input_bottomup.shape[0])
        extract_lex_error_ON1_competitor_error = np.zeros(model.sim_input_bottomup.shape[0])
        extract_lex_error_ON2_competitor_error = np.zeros(model.sim_input_bottomup.shape[0])
        extract_lex_error_ON3_competitor_error = np.zeros(model.sim_input_bottomup.shape[0])
        extract_lex_err_competitor_error = np.zeros(model.sim_input_bottomup.shape[0])
        
        extract_target_sem_error = np.zeros(model.sim_input_bottomup.shape[0])
        extract_nontarget_sem_error = np.zeros(model.sim_input_bottomup.shape[0])
        extract_sem_error_competitor_error = np.zeros(model.sim_input_bottomup.shape[0])
        extract_sem_err_competitor_error = np.zeros(model.sim_input_bottomup.shape[0])
        extract_ctx_error_competitor_error = np.zeros(model.sim_input_bottomup.shape[0])
        
        extract_ctx_error_mostactive_competitor_error = np.zeros(model.sim_input_bottomup.shape[0])
        extract_target_ctx_error = np.zeros(model.sim_input_bottomup.shape[0])
        extract_nontarget_ctx_error = np.zeros(model.sim_input_bottomup.shape[0])
        orth_overlap = np.load("./helper_txt_files/orth_overlap.npy")  # Load
        sem_overlap = np.load("./helper_txt_files/sem_overlap.npy")  # Load
        get_ON_1 = lambda target_idx : np.argsort(orth_overlap[target_idx,:])[np.sort(orth_overlap[target_idx,:]) == 1][::-1]
        get_ON_2 = lambda target_idx : np.argsort(orth_overlap[target_idx,:])[np.sort(orth_overlap[target_idx,:]) == 2][::-1]
        get_ON_3 = lambda target_idx : np.argsort(orth_overlap[target_idx,:])[np.sort(orth_overlap[target_idx,:]) == 3][::-1]
        get_SN = lambda target_idx : np.argsort(sem_overlap[target_idx,:])[np.sort(sem_overlap[target_idx,:]) > 0][::-1][1:]
        get_SF_inds = lambda target_idx : np.where(model.lexicon.semfeatmatrix[:,target_idx])[0]
        get_nontarget_SF_inds = lambda target_idx : np.delete(np.arange(model.lexicon.semfeatmatrix.shape[0]),get_SF_inds(target_idx))

        target_indices = [list(model.lexicon.words).index(w) for w in model.sim_input_bottomup]
        for input_index, target_index in enumerate(target_indices):
            temp_lex_vector_error = model.statespace['lex'][3,:,input_index].copy()
            extract_target_lex_error[input_index] = temp_lex_vector_error[target_index].copy()
            # zero out the target values
            temp_lex_vector_error[target_index] = 0
            # get most active lexical state unit competitor
            extract_lex_error_mostactive_competitor_error[input_index] = np.sort(temp_lex_vector_error)[::-1][0]
            # get ON1/2/3 competitor lex state values
            lex_error_ON1_competitors_error = temp_lex_vector_error[get_ON_1(target_indices[input_index])].sum()
            lex_error_ON2_competitors_error = temp_lex_vector_error[get_ON_2(target_indices[input_index])].sum()
            if get_ON_3(target_indices[input_index]).size !=0:
                lex_error_ON3_competitors_error = temp_lex_vector_error[get_ON_3(target_indices[input_index])].sum()
            else:
                lex_error_ON3_competitors_error = np.nan

            extract_lex_error_ON1_competitor_error[input_index] = lex_error_ON1_competitors_error
            extract_lex_error_ON2_competitor_error[input_index] = lex_error_ON2_competitors_error
            extract_lex_error_ON3_competitor_error[input_index] = lex_error_ON3_competitors_error
            temp_sem_vector_error = model.statespace['sem'][3,:,input_index].copy()
            extract_target_sem_error[input_index] = temp_sem_vector_error[get_SF_inds(target_index)].copy().sum()
            extract_nontarget_sem_error[input_index] = temp_sem_vector_error[get_nontarget_SF_inds(target_index)].sum()

        # INDICATE WHAT INFORMATION TO EXTRACT HERE
        summary_dict['extract_target_lex_state'] = extract_target_lex_state
        summary_dict['extract_lex_state_mostactive_competitor_state'] = extract_lex_state_mostactive_competitor_state
        summary_dict['extract_lex_state_ON1_competitor_state'] = extract_lex_state_ON1_competitor_state
        summary_dict['extract_lex_state_ON2_competitor_state'] = extract_lex_state_ON2_competitor_state
        summary_dict['extract_lex_state_ON3_competitor_state'] = extract_lex_state_ON3_competitor_state
        
        summary_dict['extract_target_sem_state'] = extract_target_sem_state
        summary_dict['extract_nontarget_sem_state'] = extract_nontarget_sem_state
        
        summary_dict['extract_target_ctx_state'] = extract_target_ctx_state
        summary_dict['extract_ctx_state_mostactive_competitor_state'] = extract_ctx_state_mostactive_competitor_state
        summary_dict['extract_nontarget_ctx_state'] = extract_nontarget_ctx_state
        summary_dict['extract_ctx_state_ON1_competitor_state'] = extract_ctx_state_ON1_competitor_state
        summary_dict['extract_ctx_state_ON2_competitor_state'] = extract_ctx_state_ON2_competitor_state
        summary_dict['extract_ctx_state_ON3_competitor_state'] = extract_ctx_state_ON3_competitor_state    

        summary_dict['extract_target_lex_error'] = extract_target_lex_error
        summary_dict['extract_lex_error_mostactive_competitor_error'] = extract_lex_error_mostactive_competitor_error
        summary_dict['extract_lex_error_ON1_competitor_error'] = extract_lex_error_ON1_competitor_error
        summary_dict['extract_lex_error_ON2_competitor_error'] = extract_lex_error_ON2_competitor_error
        summary_dict['extract_lex_error_ON3_competitor_error'] = extract_lex_error_ON3_competitor_error
        summary_dict['extract_target_sem_error'] = extract_target_sem_error
        summary_dict['extract_nontarget_sem_error'] = extract_nontarget_sem_error
        summary_dict['extract_target_ctx_error'] = extract_target_ctx_error
        summary_dict['extract_ctx_error_mostactive_competitor_error'] = extract_ctx_error_mostactive_competitor_error
        summary_dict['extract_nontarget_ctx_error'] = extract_nontarget_ctx_error

    if model.get_all_values:
        print('Warning! We are collecting all state activity for every input')
        summary_dict['full_ctx_state'] =  model.statespace['ctx'][0]
        summary_dict['full_sem_state'] =  model.statespace['sem'][0]
        summary_dict['full_sem_error'] =  model.statespace['sem'][3]
        summary_dict['full_lex_state'] =  model.statespace['lex'][0]
        summary_dict['full_lex_error'] =  model.statespace['lex'][3]
        
    summary_dict['max_lex_state_activation'] =  np.max(model.statespace['lex'][0], axis = 0)
    summary_dict['total_lex_state'] = np.sum(model.statespace['lex'][0], axis = 0) 
    summary_dict['total_lexsem_err'] = np.sum(model.statespace['lex'][3], axis = 0) + np.sum(model.statespace['sem'][3], axis = 0)
    summary_dict['total_lex_err'] = np.sum(model.statespace['lex'][3], axis = 0) 
    summary_dict['total_sem_err'] = np.sum(model.statespace['sem'][3], axis = 0)
    summary_dict['max_lex_state_identity'] =  np.argmax(model.statespace['lex'][0], axis = 0)
    summary_dict['secmax_lex_state_activation'] =  np.sort(model.statespace['lex'][0], axis = 0)[-2,:]


    return summary_dict