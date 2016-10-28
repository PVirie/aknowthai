import tensorflow as tf
import numpy as np


def final_layer(lstm_output, w):
    return tf.matmul(lstm_output, w)


def split_output(final_output, batches, total_classes):
    """ (batch, [one-hots, alpha]) ->  (batch, 1), (batch, one-hots)"""
    alphas = tf.slice(final_output, [0, total_classes], [batches, 1])
    logits = tf.slice(final_output, [0, 0], [batches, total_classes])
    return alphas, logits


def logit_to_label(logits):
    """ activated labels must be forced sparse! """
    return tf.nn.softmax(logits)


def make_prob(preAlphas):
    """ preAlphas will be normalized into range [0, 1] """
    return tf.sigmoid(preAlphas)


def build_shifting_graph(alphas, identity, shift_template):
    return identity + tf.mul(tf.reshape(alphas, [-1, 1, 1]), shift_template)


def build_step_cost_function(c0, shift, alphas, weights, logits, true_labels, batches, total_characters, total_classes):
    """ total_characters = true_character_number + 1, the last slot is for terminal penalty """
    """ true_labels of integer type can be expaned into [batches, total_characters, total_classes] via tf.one_hot """
    """ logits [batches, total_classes] """
    """ weights [batches, total_characters, 1] """
    """ alphas [batches, 1]  """
    """ shift [batches, total_characters, total_characters] """
    """ cost for each item in batch = \sum_i {c_0*(1-\alpha_i) + \alpha_i.prediction_error} + c_1*total_weight_at_last_step  """
    """ c_1 is some high value, 1 >= c_0 >= (total_characters - num_characters_in_item)/total_characters """

    new_weights = tf.batch_matmul(shift, weights)
    y = tf.reshape(tf.one_hot(true_labels, depth=total_classes, dtype=tf.float32, axis=-1), [batches, total_characters, total_classes])
    z = tf.reshape(tf.nn.log_softmax(logits), [batches, 1, total_classes])
    delta = tf.mul(new_weights, -tf.mul(y, z))
    cost = tf.reduce_sum(delta) + tf.reduce_sum(1 - alphas) * c0
    return new_weights, cost


def build_tailing_cost_function(c1, weights, batches, total_characters):
    return tf.reduce_sum(tf.slice(weights, [0, total_characters - 1, 0], [batches, 1, 1])) * c1


class Network:

    def __init__(self, data_shape, labels_shape, total_classes):
        self.sess = tf.Session()
        total_characters = labels_shape[1]
        batches = data_shape[0]
        input_size = data_shape[2]
        input_sequence_length = data_shape[1]
        self.gpu_inputs = tf.placeholder(tf.float32, data_shape)
        self.gpu_labels = tf.placeholder(tf.int32, labels_shape)
        self.sharpeness = tf.placeholder(tf.float32)

        lstm = tf.nn.rnn_cell.BasicLSTMCell(input_size, state_is_tuple=False)
        self.stacked_lstm = tf.nn.rnn_cell.MultiRNNCell([lstm] * 2, state_is_tuple=False)

        r12340 = np.arange(1, total_characters + 1, 1, dtype=np.int32)
        r12340[total_characters - 1] = 0
        shift = -np.identity(total_characters) + np.identity(total_characters)[:, r12340]
        shift[0, total_characters - 1] = 0
        shift[total_characters - 1, total_characters - 1] = 0

        w = np.zeros((batches, total_characters, 1))
        w[:, 0] = 1.0
        weights = tf.Variable(w, trainable=False, dtype=tf.float32)

        W = tf.Variable(np.random.rand(input_size, total_classes + 1) * 0.1, dtype=tf.float32)

        state = self.stacked_lstm.zero_state(data_shape[0], tf.float32)
        output = tf.placeholder(dtype=tf.float32, shape=[batches, self.stacked_lstm.output_size])

        sum_cost = tf.Variable(0, trainable=False, dtype=tf.float32)
        with tf.variable_scope("lstm") as scope:
            for i in range(input_sequence_length):
                if i > 0:
                    scope.reuse_variables()
                output, state = self.stacked_lstm(tf.reshape(tf.slice(self.gpu_inputs, [0, i, 0], [batches, 1, input_size]), [batches, input_size]), state)
                preAlphas, logits = split_output(final_layer(output, W), batches, total_classes)
                alphas = make_prob(tf.mul(preAlphas, self.sharpeness))
                shift_mat = build_shifting_graph(alphas, tf.constant(np.identity(total_characters), dtype=tf.float32), tf.constant(shift, dtype=tf.float32))
                weights, cost = build_step_cost_function(1.0, shift_mat, alphas, weights, logits, self.gpu_labels, batches, total_characters, total_classes)
                sum_cost = sum_cost + cost

        self.overall_cost = sum_cost
        # + build_tailing_cost_function(10000.0, weights, batches, total_characters)

        self.training_op = tf.train.AdagradOptimizer(0.001).minimize(self.overall_cost)

        self.saver = tf.train.Saver()

        # Before starting, initialize the variables.  We will 'run' this first.
        init = tf.initialize_all_variables()

        # Launch the graph.
        self.sess.run(init)

    def train(self, data, labels, session_name, max_iteration):
        for step in xrange(max_iteration):
            _, loss = self.sess.run((self.training_op, self.overall_cost), feed_dict={self.gpu_inputs: data, self.gpu_labels: labels, self.sharpeness: [step * 1.0]})
            print loss
        self.saver.save(self.sess, "../artifacts/" + session_name)

    def load_session(self, session_name):
        self.saver.restore(self.sess, "../artifacts/" + session_name)

    def eval(self, data):
