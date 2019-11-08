import tensorflow as tf
from layers import bioRNN_Cell
from parameters import par

def build_model(n_hidden, n_output, EI=True, excitatory_frac=0.8,
    balance_EI=True, connection_prob=1., synapse_config='full', n_receptive_fields=1.,
    dt=10, tau_slow=1500., tau_fast=200., membrane_time_constant=100.,
    noise_rnn_sd=0.5, **kwargs):
    """
    Builds a Keras model using the Functional API which uses bioRNN_Cell
    """
    obs_input = tf.keras.layers.Input(shape=[par['n_input'],], name='obs_input')
    h_input = tf.keras.layers.Input(shape=[n_hidden,], name='h_input')
    syn_x_input = tf.keras.layers.Input(shape=[n_hidden,], name='syn_x_input')
    syn_u_input = tf.keras.layers.Input(shape=[n_hidden,], name='syn_u_input')
    logits, [h, syn_x, syn_u] = bioRNN_Cell(n_hidden, n_output, EI=EI, excitatory_frac=excitatory_frac,
        balance_EI=balance_EI, connection_prob=connection_prob, synapse_config=synapse_config,
        n_receptive_fields=n_receptive_fields, dt=dt, tau_slow=tau_slow, tau_fast=tau_fast,
        membrane_time_constant=membrane_time_constant, noise_rnn_sd=noise_rnn_sd,
        name='rnn')(obs_input, [h_input, syn_x_input, syn_u_input])
    return tf.keras.Model(inputs=[obs_input, h_input, syn_x_input, syn_u_input],
        outputs=[logits, h, syn_x, syn_u])

@tf.function
def do_trial(model, x):
    """
    x has shape (Tsteps, Batch, n_input)
    """
    tsteps = x.shape[0]
    batch_size = x.shape[1]
    n_hidden = model.get_layer('h_input').output_shape[0][1]
    h = tf.zeros([batch_size, n_hidden])
    syn_x = tf.ones([batch_size, n_hidden])
    syn_u = tf.stack([tf.squeeze(tf.constant(model.get_layer('rnn').U)) for i in range(batch_size)])

    logits_list = []
    h_list = []
    for obs in tf.unstack(x):
        logits, h, syn_x, syn_u = model([obs, h, syn_x, syn_u])
        logits_list.append(logits)
        h_list.append(h)
    logits_seq = tf.stack(logits_list)
    h_seq = tf.stack(h_list)
    return logits_seq, h_seq
