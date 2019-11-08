import tensorflow as tf
from parameters import par


def compute_loss(y_true, y_logits, h=None, w_rnn=None, mask=None, loss_norm=2,
    spike_cost=1e-2, weight_cost=0.):
    """
    - y_true, y_logits have shape (Tsteps, Batch, n_pol)
    - mask has shape (Tsteps, Batch)
    """
    if mask is None: mask = tf.ones([y_true.shape[0], y_true.shape[1], 1])
    xe_loss = tf.reduce_mean(mask*tf.nn.softmax_cross_entropy_with_logits(
        y_true, y_logits, axis=-1))

    if h is not None:
        spike_loss = tf.reduce_mean(h**loss_norm)
    else:
        spike_loss = 0.

    if w_rnn is not None:
        weight_loss = tf.reduce_mean(tf.nn.relu(w_rnn)**loss_norm)
    else:
        weight_loss = 0.

    len_mask = tf.math.count_nonzero(mask[:,0])
    preds = tf.math.argmax(y_logits, axis=2) # (T, B)
    preds = tf.one_hot(preds, par['n_output'], axis=-1) # (T, B, n_pol)
    mask = tf.expand_dims(mask, axis=-1)
    count_true = tf.math.count_nonzero(mask*preds*y_true, axis=-1) # (T, B)
    acc = tf.reduce_mean(tf.reduce_sum(count_true, axis=0)/len_mask)
    return xe_loss + spike_cost*spike_loss + weight_cost*weight_loss, acc
