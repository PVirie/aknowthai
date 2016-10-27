import tensorflow as tf
import numpy as np


def split_output(lstm_output, batches, total_classes):
    """ (batch, [one-hots, alpha]) ->  (batch, 1), (batch, one-hots)"""
    alphas = tf.slice(lstm_output, [0, total_classes], [batches, 1])
    logits = tf.slice(lstm_output, [0, 0], [batches, total_classes])
    return alphas, logits


def logit_to_label(logits):
    """ activated labels must be forced sparse! """
    return tf.nn.softmax(logits)


def build_shifting_graph(alphas, identity, shift_template):
    return identity + tf.mul(tf.reshape(alphas, [-1, 1, 1]), shift_template)


def build_step_cost_function(c0, shift, alphas, weights, logits, true_labels, batches, total_characters, total_classes):
    """ total_characters = true_character_number + 1, the last slot is for terminal penalty """
    """ true_labels of integer type can be expaned into [total_characters, total_classes] via tf.one_hot """
    """ logits [batches, total_classes] """
    """ weights [batches, total_characters, 1] """
    """ alphas [batches, 1]  """
    """ shift [batches, total_characters, total_characters] """
    """ cost for each item in batch = \sum_i {c_0*(1-\alpha_i) + \alpha_i.prediction_error} + c_1*total_weight_at_last_step  """
    """ c_1 is some high value, 1 >= c_0 >= (total_characters - num_characters_in_item)/total_characters """

    new_weights = tf.batch_matmul(shift, weights)
    y = tf.reshape(tf.one_hot(true_labels, depth=total_classes, dtype=tf.float32, axis=-1), [1, total_characters, total_classes])
    z = tf.reshape(tf.nn.log_softmax(logits), [batches, 1, total_classes])
    delta = tf.mul(new_weights, -tf.mul(y, z))
    cost = tf.reduce_sum(delta) + tf.reduce_sum(1 - alphas) * c0
    return new_weights, cost


def build_tailing_cost_function(c1, weights, batches, total_characters):
    return tf.reduce_sum(tf.slice(weights, [0, total_characters - 1, 0], [batches, 1, 1])) * c1


lstm = tf.nn.rnn_cell.BasicLSTMCell(10)
state = tf.zeros([20, lstm.state_size])
input = tf.zeros([20, 10])
print lstm.state_size
output, state = lstm(input, state)

shift = -np.identity(5) + np.identity(5)[:, [1, 2, 3, 4, 0]]
shift[0, 4] = 0
shift[4, 4] = 0

w = np.zeros((20, 5, 1))
w[:, 0] = 1.0
weights = tf.Variable(w, trainable=False, dtype=tf.float32)

true_labels = tf.constant([0, 1, 2, 3, -1], dtype=tf.int32)

alphas, labels = split_output(tf.constant(np.array(range(200)) / 200.0, shape=[20, 10], dtype=tf.float32), 20, 9)
shift_mat = build_shifting_graph(alphas, tf.constant(np.identity(5), dtype=tf.float32), tf.constant(shift, dtype=tf.float32))
new_weights, cost = build_step_cost_function(1.0, shift_mat, alphas, weights, labels, true_labels, 20, 5, 9)
tailing_cost = build_tailing_cost_function(100.0, new_weights, 20, 5)

# Before starting, initialize the variables.  We will 'run' this first.
init = tf.initialize_all_variables()

# Launch the graph.
sess = tf.Session()
sess.run(init)


print sess.run(cost + tailing_cost)
