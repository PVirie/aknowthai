import tensorflow as tf
import numpy as np


def split_output(lstm_output, batches, total_classes):
    """ (batch, [one-hots, alpha]) ->  """
    alphas = tf.slice(lstm_output, [0, total_classes], [batches, 1])
    labels = tf.slice(lstm_output, [0, 0], [batches, total_classes])
    return alphas, labels


def build_shifting_graph(alphas, identity, shift_template):
    return identity + tf.mul(tf.reshape(alphas, [-1, 1, 1]), shift_template)


def build_cost_function(shift, alphas, weights, one_hot_labels, true_labels, batches, total_characters, total_classes):
    """ total_characters = true_character_number + 1, the last slot is for terminal penalty """
    """ true_labels of int type can be expaned into [total_characters, total_classes] via tf.one_hot """
    """ one_hot_labels [batches, total_classes] is NOT one hots """
    """ weights [batches, total_characters, 1] is NOT one hots """
    """ alphas [batches, 1] is NOT one hots """
    """ shift [batches, total_characters, total_characters] is NOT one hots """

    new_weights = tf.batch_matmul(shift, weights)
    y = tf.reshape(tf.one_hot(true_labels, depth=total_classes, dtype=tf.float32, axis=-1), [1, total_characters, total_classes])
    z = tf.reshape(one_hot_labels, [batches, 1, total_classes])
    delta = tf.mul(new_weights, tf.square(tf.sub(y, z)))
    cost = tf.reduce_sum(delta)
    return new_weights, cost


lstm = tf.nn.rnn_cell.BasicLSTMCell(10)
state = tf.zeros([20, lstm.state_size])
input = tf.zeros([20, 10])
print lstm.state_size
output, state = lstm(input, state)

shift = -np.identity(5) + np.identity(5)[:, [1, 2, 3, 4, 0]]
shift[0, 4] = 0
shift[4, 4] = 0
print shift

w = np.zeros((20, 5, 1))
w[:, 0] = 1.0
print w
weights = tf.Variable(w, trainable=False, dtype=tf.float32)

true_labels = tf.constant([0, 1, 2, 3, -1], dtype=tf.int32)

alphas, labels = split_output(tf.constant(np.array(range(200)) / 200.0, shape=[20, 10], dtype=tf.float32), 20, 9)
shift_mat = build_shifting_graph(alphas, tf.constant(np.identity(5), dtype=tf.float32), tf.constant(shift, dtype=tf.float32))
new_weights, cost = build_cost_function(shift_mat, alphas, weights, labels, true_labels, 20, 5, 9)

# Before starting, initialize the variables.  We will 'run' this first.
init = tf.initialize_all_variables()

# Launch the graph.
sess = tf.Session()
sess.run(init)


print sess.run(cost)
