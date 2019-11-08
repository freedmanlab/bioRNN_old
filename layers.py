import tensorflow as tf
import numpy as np

class bioRNN_Cell(tf.keras.layers.AbstractRNNCell):
    def __init__(self, n_hidden, n_output, EI=True, excitatory_frac=0.8,
        balance_EI=True, connection_prob=1., synapse_config='full', n_receptive_fields=1.,
        dt=10, tau_slow=1500., tau_fast=200., membrane_time_constant=100.,
        noise_rnn_sd=0.5, **kwargs):
        self.units = n_hidden
        super(bioRNN_Cell, self).__init__(**kwargs)

        # Copying over args
        self.EI = EI
        self.balance_EI = balance_EI
        self.connection_prob = connection_prob
        self.synapse_config = synapse_config
        self.n_hidden = n_hidden
        self.n_output = n_output

        # Stuff computed from args
        self.dt_sec = tf.constant(dt/1000)
        self.alpha_neuron = tf.constant(dt/membrane_time_constant)
        self.noise_rnn = tf.math.sqrt(2*self.alpha_neuron)*noise_rnn_sd
        # self.noise_in =
        self.input_connection_prob = connection_prob/n_receptive_fields

        # EI stuff
        self.num_exc_units = tf.math.round(excitatory_frac*n_hidden)
        self.num_inh_units = n_hidden - self.num_exc_units
        self.EI_list = [1. for i in range(self.num_exc_units)] + [-1. for i in range(self.num_inh_units)]
        self.EI_matrix = tf.linalg.diag(tf.constant(self.EI_list))

        # Synaptic plasticity stuff
        synapse_config_list = self._get_synapse_config_list(synapse_config)
        self.alpha_stf = np.ones([1, n_hidden], dtype=np.float32)
        self.alpha_std = np.ones([1, n_hidden], dtype=np.float32)
        self.U = np.ones([1, n_hidden], dtype=np.float32)
        self.dynamic_synapse = np.zeros([1, n_hidden], dtype=np.float32)
        for i in range(n_hidden):
            if synapse_config_list[i] is 'facilitating':
                self.alpha_stf[0,i] = dt/tau_slow
                self.alpha_std[0,i] = dt/tau_fast
                self.U[0,i] = 0.15
                self.dynamic_synapse[0,i] = 1.
            elif synapse_config_list[i] is 'depressing':
                self.alpha_stf[0,i] = dt/tau_fast
                self.alpha_std[0,i] = dt/tau_slow
                self.U[0,i] = 0.45
                self.dynamic_synapse[0,i] = 1.
            elif synapse_config_list[i] is 'static':
                # If static, leave at default.
                pass

    @property
    def state_size(self):
        return self.units

    def _get_synapse_config_list(self, synapse_config):
        _dict = {
            'full': ['facilitating' if i%2==0 else 'depressing' for i in range(self.n_hidden)],
            'fac': ['facilitating' for i in range(self.n_hidden)],
            'dep': ['depressing' for i in range(self.n_hidden)]
            # 'exc_fac': ['facilitating' if self.EI_list==1 else 'static' for i in range(self.n_hidden)],
            # 'exc_dep': None,
            # 'inh_fac': None,
            # 'inh_dep': None,
            # 'exc_dep_inh_fac': None
        }
        return _dict[synapse_config]

    def _sample_weights(self, dims, connection_prob, shape_param=0.1, scale_param=1.0):
        """
        Sample weights from Gamma distribution, then prune according to
        connection_prob.

        - dims: [num_row, num_col] for weight matrix
        - connection_prob: scalar in [0,1]
        - shape_param, scale_param are parameters for the Gamma distribution
        """
        w_ = np.random.gamma(shape_param, scale=scale_param, size=dims)
        prune_mask = (np.random.uniform(size=dims) < connection_prob)
        return w_ * prune_mask

    def build(self, input_shape):
        """
        This is called under the hood when Keras uses this layer in a model.
        input_shape is figured out automatically.
        """
        _w_in = self._sample_weights([input_shape[-1], self.n_hidden],
            self.input_connection_prob, shape_param=0.2)

        _w_out = self._sample_weights([self.n_hidden, self.n_output], self.connection_prob)
        inh_mask = np.ones_like(_w_out)
        inh_mask[-self.num_inh_units:, :] = 0.
        _w_out *= inh_mask

        _w_rnn = self._sample_weights([self.n_hidden, self.n_hidden], self.connection_prob)
        if self.balance_EI:
            _w_rnn[:, -self.num_inh_units:] = \
                self._sample_weights([self.n_hidden, self.num_inh_units], self.connection_prob,
                    shape_param=0.2)
            _w_rnn[-self.num_inh_units:, :] = \
                self._sample_weights([self.num_inh_units, self.n_hidden], self.connection_prob,
                    shape_param=0.2)
        self_connections_prune_mask = np.ones_like(_w_rnn) - np.eye(self.n_hidden)
        _w_rnn *= self_connections_prune_mask

        self.W_in = tf.Variable(_w_in, trainable=True, dtype=tf.float32, name='W_in')
        self.W_rnn = tf.Variable(_w_rnn, trainable=True, dtype=tf.float32, name='W_rnn')
        self.b_rnn = tf.Variable(tf.zeros([1, self.n_hidden]), trainable=True, dtype=tf.float32, name='b_rnn')
        self.W_out = tf.Variable(_w_out, trainable=True, dtype=tf.float32, name='W_out')
        self.b_out = tf.Variable(tf.zeros([1, self.n_output]), trainable=True, dtype=tf.float32, name='b_out')
        self.built = True

    def call(self, input, state):
        """
        Update synaptic plasticity if on.
        Update and return hidden state (and synaptic plasticity vals).

        - Input has shape [batch, n_input]
        - Each state matrix has shape [batch_size, n_hidden].
        """
        h, syn_x, syn_u = state
        if self.synapse_config is not None:
            syn_x += self.dynamic_synapse*(self.alpha_std*(1-syn_x) - self.dt_sec*syn_u*syn_x*h)
            syn_u += self.dynamic_synapse*(self.alpha_stf*(self.U-syn_u) + self.dt_sec*self.U*(1-syn_u)*h)
            syn_x = tf.math.minimum(1., tf.nn.relu(syn_x))
            syn_u = tf.math.minimum(1., tf.nn.relu(syn_u))
            h_post = syn_u * syn_x * h
        else:
            h_post = h

        first_part = (1-self.alpha_neuron)*h
        second_part = self.alpha_neuron*(input @ tf.nn.relu(self.W_in)
            + h_post @ self.EI_matrix @ tf.nn.relu(self.W_rnn) + self.b_rnn)
        third_part = tf.random.normal(tf.shape(h), mean=0., stddev=self.noise_rnn)
        h = tf.nn.relu(first_part + second_part + third_part)
        output = h @ tf.nn.relu(self.W_out) + self.b_out
        return output, [h, syn_x, syn_u]
