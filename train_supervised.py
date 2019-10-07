import tensorflow as tf
from model import do_trial
from loss import compute_loss
from parameters import par
import stimulus

@tf.function
def train_step(model, opt, x, y_true, mask=None):
    """
    Args:
    - model: Keras model
    - x: (T, B, dim)
    - y_true: (T, B, dim)
    """
    with tf.GradientTape() as tape:
        logits_seq, h_seq = do_trial(model, x)
        loss, acc = compute_loss(y_true, logits_seq, h=h_seq, mask=mask)
    grads = tape.gradient(loss, model.trainable_variables)
    opt.apply_gradients(zip(grads, model.trainable_variables))
    return loss, acc


def train_loop(model, opt, task_name_or_index, n_iters):
    task_dict = {
        'go': 0,
        'rt_go': 1,
        'dly_go': 2,
        'anti-go': 3,
        'anti-rt_go': 4,
        'anti-dly_go': 5,
        'dm1': 6,
        'dm2': 7,
        'ctx_dm1': 8,
        'ctx_dm2': 9,
        'multsen_dm': 10,
        'dm1_dly': 11,
        'dm2_dly': 12,
        'ctx_dm1_dly': 13,
        'ctx_dm2_dly': 14,
        'multsen_dm_dly': 15,
        'dms': 16,
        'dmc': 17,
        'dnms': 18,
        'dnmc': 19
        }

    if isinstance(task_name_or_index, str):
        task_index = task_dict[task_name_or_index]
    else:
        task_index = task_name_or_index

    stim = stimulus.MultiStimulus()

    for i in range(n_iters):
        """
        # add sanity checks here
        if i%50 == 0:
            w_rnn = model.get_layer('rnn').W_rnn
            pct_nonzero_weights = tf.math.count_nonzero(w_rnn) / (w_rnn.shape[0]*w_rnn.shape[1])
            diagonal_sum = tf.reduce_sum([w_rnn[i,i] for i in range(tf.math.reduce_min([w_rnn.shape[0], w_rnn.shape[1]]))])
            tf.print('diag sum: ', diagonal_sum)
        """

        name, input_data, ytrue_data, dead_time_mask, reward_data = \
            stim.generate_trial(task_index)
        input_data = tf.constant(input_data, dtype=tf.float32)
        ytrue_data = tf.constant(ytrue_data, dtype=tf.float32)
        dead_time_mask = tf.constant(dead_time_mask, dtype=tf.float32)
        loss, acc = train_step(model, opt, input_data, ytrue_data, mask=dead_time_mask)
        if i%50 == 0: tf.print('Iter: ', i, '| Loss: ', loss, '| Acc: ', acc, '\n')
