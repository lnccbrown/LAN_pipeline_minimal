# Append system path to include the config scripts
import sys
import os
from copy import deepcopy

print('importing lanfactory')
import lanfactory

print('importing ssms')
import ssms

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from config import *
from config import model_performance_utils

import torch
import pandas as pd
import numpy as np

import pickle
import yaml

def make_data_generator_configs(model = 'ddm',
                                generator_approach = 'lan',
                                data_generator_arg_dict = None,
                                model_config_arg_dict = None,
                                save_name = None,
                                save_folder = ''):
    
    # Load copy of the respective model's config dict from ssms
    model_config = deepcopy(ssms.config.model_config[model.split('_deadline')[0]])
    
    # Load copy of the respective data_generator_config dicts 
    data_config = deepcopy(ssms.config.data_generator_config[generator_approach])
    data_config['model'] = model
    
    for key, val in data_generator_arg_dict.items():
        data_config[key] = val
        
    for key, val in model_config_arg_dict.items():
        model_config[key] = val

    config_dict = {'model_config': model_config, 'data_config': data_config}
    
    if save_name is not None:
        if len(save_folder) > 0:
            
            if save_folder[-1] == '/':
                pass
            else:
                save_folder = save_folder + '/'
        
        # Create save_folder if not already there
        lanfactory.utils.try_gen_folder(folder = save_folder, 
                                        allow_abs_path_folder_generation = True)
                
        # Dump pickle file
        pickle.dump(config_dict, open(save_folder + save_name, 'wb'))
        
        print('Saved to: ')
        print(save_folder + save_name)
    
    return {'config_dict':config_dict, 
            'config_file_name': None if save_name is None else save_folder + save_name}

def get_data_generator_config(yaml_config_path = None,
                              base_path = None):

    basic_config = yaml.safe_load(open(yaml_config_path, 'rb'))
    
    training_data_folder = base_path + 'data/training_data/' + basic_config['GENERATOR_APPROACH'] + \
                            '/training_data_n_samples_' + \
                                str(basic_config['N_SAMPLES']) + '_dt_' + str(basic_config['DELTA_T']) + '/' + \
                                    str(basic_config['MODEL']) + '/'
           
    data_generator_arg_dict = {'output_folder': training_data_folder,
                               'model': basic_config['MODEL'],
                               'n_samples': basic_config['N_SAMPLES'],
                               'n_parameter_sets': basic_config['N_PARAMETER_SETS'],
                               'delta_t': basic_config['DELTA_T'],
                               'n_training_samples_by_parameter_set': basic_config['N_TRAINING_SAMPLES_BY_PARAMETER_SET'],
                               'n_subruns': basic_config['N_SUBRUNS'],
                               'cpn_only': True if (basic_config['GENERATOR_APPROACH'] == 'cpn') else False}

    model_config_arg_dict = {}

    config_dict = make_data_generator_configs(model = basic_config['MODEL'],
                                              generator_approach = basic_config['GENERATOR_APPROACH'],
                                              data_generator_arg_dict = data_generator_arg_dict,
                                              model_config_arg_dict = model_config_arg_dict,
                                              save_name = None,
                                              save_folder = None)
    return config_dict

    
def make_train_network_configs(training_data_folder = None,
                               train_val_split = 0.9, 
                               save_folder = '',
                               network_arg_dict = None,
                               train_arg_dict = None,
                               save_name = None):
    
    # Load 
    train_config = deepcopy(lanfactory.config.train_config_mlp)
    network_config = deepcopy(lanfactory.config.network_config_mlp)
    
    for key, val in network_arg_dict.items():
        network_config[key] = val
        
    for key, val in train_arg_dict.items():
        train_config[key] = val
    
    config_dict = {'network_config': network_config,
                   'train_config': train_config,
                   'training_data_folder': training_data_folder,
                   'train_val_split': train_val_split}
    
    if save_name is not None:
        if len(save_folder) > 0:
            
            if save_folder[-1] == '/':
                pass
            else:
                save_folder = save_folder + '/'
        
        # Create save_folder if not already there
        lanfactory.utils.try_gen_folder(folder = save_folder, 
                                        allow_abs_path_folder_generation = True)
             
        # Dump pickle file
        print('Saved to: ')
        print(save_folder + save_name)
        
        pickle.dump(config_dict, open(save_folder + save_name, 'wb'))
    
    return {
            'config_dict': config_dict, 
            'config_file_name': None if save_name is None else save_folder + save_name
           }

def get_train_network_config(yaml_config_path = None,
                             net_index = 0):

    basic_config = yaml.safe_load(open(yaml_config_path, 'rb'))
    network_type = basic_config['NETWORK_TYPE']

    # Train output type specifies what the network output node
    # 'represents' (e.g. log-probabilities / logprob, logits, probabilities / prob)

    # Specifically for cpn, we train on logit outputs for numerical stability, then transform outputs
    # to log-probabilities when running the model in evaluation / inference mode 
    train_output_type_dict = {'lan': 'logprob',
                              'cpn': 'logits',
                              'opn': 'logits',
                              'gonogo': 'logits',
                              'cpn_bce': 'prob'}

    # Last layer activation depending on train output type
    output_layer_dict = {'logits': 'linear',
                         'logprob': 'linear',
                         'prob': 'sigmoid'}

    # LOSS 
    # 'bce' (for binary-cross-entropy), use when train output is 'prob'
    # 'bcelogit' (for binary-cross-entropy with inputs representing logits) use when train output type is 'logits', (this is standard for cpns)
    # 'huber' (usually) used when train output is 'logprob'

    train_loss_dict = {'logprob': 'huber',
                       'logits': 'bcelogit',
                       'prob': 'bce'
                       }

    data_key_dict = {'lan': {'features_key': 'lan_data', 
                             'label_key': 'lan_labels'},
                     'cpn': {'features_key': 'cpn_data',
                             'label_key': 'cpn_labels'},
                     'opn': {'features_key': 'opn_data',
                             'label_key': 'opn_labels'},
                     'gonogo': {'features_key': 'gonogo_data',
                               'label_key': 'gonogo_labels'},
                     }

    # Network architectures
    layer_sizes = basic_config['LAYER_SIZES'][net_index]
    activations = basic_config['ACTIVATIONS'][net_index]
    activations.append(output_layer_dict[train_output_type_dict[network_type]])
    # Append last layer (type of layer depends on type of network as per train_output_type_dict dictionary above)

    # Number is set to 10000 here (an upper bound), for training on all available data (usually roughly 300 files, but has never been more than 1000)
    # For numerical experiments, one may want to artificially constraint the number of training files to teest the impact on network performance
  
    network_arg_dict = {'train_output_type': train_output_type_dict[network_type],
                        'network_type': network_type}

    network_arg_dict['layer_sizes'] = layer_sizes
    network_arg_dict['activations'] = activations
    
    # initial train_arg_dict
    # refined in for loop in next cell
    train_arg_dict = {'n_epochs': basic_config['N_EPOCHS'],
                      'loss': train_loss_dict[train_output_type_dict[network_type]],
                      'optimizer': basic_config['OPTIMIZER_'],
                      'train_output_type': train_output_type_dict[network_type],
                      'n_training_files': basic_config['N_TRAINING_FILES'],
                      'train_val_split': basic_config['TRAIN_VAL_SPLIT'],
                      'weight_decay': basic_config['WEIGHT_DECAY'],
                      'cpu_batch_size': basic_config['CPU_BATCH_SIZE'],
                      'gpu_batch_size': basic_config['GPU_BATCH_SIZE'],
                      'shuffle_files': basic_config['SHUFFLE'],
                      'label_lower_bound': eval(basic_config['LABELS_LOWER_BOUND']),
                      'layer_sizes': layer_sizes,
                      'activations': activations,
                      'learning_rate': basic_config['LEARNING_RATE'],
                      'features_key': data_key_dict[network_type]['features_key'],
                      'label_key': data_key_dict[network_type]['label_key'],
                      'save_history': True,
                      'lr_scheduler': basic_config['LR_SCHEDULER'],
                      'lr_scheduler_params': basic_config['LR_SCHEDULER_PARAMS']
                      }

    config = make_train_network_configs(training_data_folder = basic_config['TRAINING_DATA_FOLDER'],
                                        train_val_split = basic_config['TRAIN_VAL_SPLIT'],
                                        save_name = None,
                                        train_arg_dict = train_arg_dict,
                                        network_arg_dict = network_arg_dict,
                                        )
    # Add some extra fields to our config dictionary (other scripts might need these)
    config['extra_fields'] = {'model': basic_config['MODEL']}
    
    return config
          
def make_param_recovery_configs(model_name = 'ddm',
                                parameter_recovery_data_loc = '',
                                lan_files = [],
                                lan_config_files = [],
                                lan_ids = [],
                                save_folder = '',
                                model_config = None,
                                n_burn = 1000,
                                n_mcmc = 5000,
                                n_chains = 4):
    
    parameter_recovery_config_dict = {}
    parameter_recovery_config_dict['parameter_recovery_data_loc'] = parameter_recovery_data_loc
    parameter_recovery_config_dict['model_config'] = model_config
    parameter_recovery_config_dict['model_name'] = model_name
    parameter_recovery_config_dict['lan_files'] = lan_files
    parameter_recovery_config_dict['lan_ids'] = lan_ids
    parameter_recovery_config_dict['lan_config_files'] = lan_config_files
    parameter_recovery_config_dict['n_burn'] = n_burn
    parameter_recovery_config_dict['n_mcmc'] = n_mcmc
    parameter_recovery_config_dict['n_chains'] = n_chains
    parameter_recovery_config_dict['save_folder'] = save_folder
    parameter_recovery_config_dict['save_file'] = save_folder + '/' + model_name + \
                     '_parameter_recovery_run_config.pickle'
    
    lanfactory.utils.try_gen_folder(folder = save_folder, 
                                    allow_abs_path_folder_generation = True)
    
    pickle.dump(parameter_recovery_config_dict, 
                open(save_folder + '/' + model_name + '_parameter_recovery_run_config.pickle', 'wb'))
    
    print('Saving to: ')
    print(parameter_recovery_config_dict['save_file'])
    
    return parameter_recovery_config_dict