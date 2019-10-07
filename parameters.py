import numpy as np
import tensorflow as tf

global par

par = {
	# train params
	'training_method': 'RL',
	'batch_size': 256, # 256

	# RL params
	'fix_break_penalty': -1.,
	'wrong_choice_penalty': -0.01,
	'correct_choice_reward': 1.,

	 # Task specs
    'task'                  : 'multistim',  # See stimulus file for more options
    'n_tasks'               : 20,
    'multistim_trial_length': 2000,
    'dt'					: 20,  # 20
    'mask_duration'         : 0,
    'dead_time'             : 200,

    # Tuning function data
    'num_motion_dirs'       : 8,
    'tuning_height'         : 4.0,          # von Mises magnitude scaling factor

    # Network shape
    'num_motion_tuned'      : 64,
    'num_fix_tuned'         : 4,
    'num_rule_tuned'        : 0,
    'include_rule_signal'   : True,
    'n_hidden': 256, # change for debug

    # Variance values
    'clip_max_grad_val'     : 1.0,
    'input_mean'            : 0.0,
    'noise_in_sd'           : 0.0,
    'noise_rnn_sd'          : 0.05,
    'membrane_time_constant': 100,

    # Gating
    'gating_type': 'XdG',
    'gate_pct': 0.8

    }


def update_dependencies():

	# Number of output neurons
    par['n_output'] = par['num_motion_dirs'] + 1
    par['n_pol'] = par['num_motion_dirs'] + 1

    # Number of input neurons
    par['n_input'] = par['num_motion_tuned'] + par['num_fix_tuned'] + par['num_rule_tuned']

    # Set trial step length
    par['num_time_steps'] = par['multistim_trial_length']//par['dt']

    # Specify time step in seconds and neuron time constant
    par['dt_sec'] = par['dt']/1000
    par['alpha_neuron'] = np.float32(par['dt'])/par['membrane_time_constant']

    # Generate noise deviations
    par['noise_rnn'] = np.sqrt(2*par['alpha_neuron'])*par['noise_rnn_sd']
    par['noise_in'] = np.sqrt(2/par['alpha_neuron'])*par['noise_in_sd']


def gen_gating():
    """
    Generate the gating signal to applied to all hidden units
    """
    par['gating'] = []

    for t in range(par['n_tasks']):
        gating_task = np.zeros(par['n_hidden'], dtype=np.float32)
        for i in range(par['n_hidden']):

            if par['gating_type'] == 'XdG':
                if np.random.rand() < 1-par['gate_pct']:
                    gating_task[i] = 1

            elif par['gating_type'] == 'split':
                if t%par['n_subnetworks'] == i%par['n_subnetworks']:
                    gating_layer[i] = 1

            elif par['gating_type'] is None:
                gating_task[i] = 1

        par['gating'].append(gating_task)


update_dependencies()
# gen_gating()

"""
if __name__ == '__main__':
	gates = np.stack([par['gating'][task] for task in range(par['n_tasks'])])
	print(gates.shape)
	np.save('./gates/20gates_2.npy', gates)
"""
