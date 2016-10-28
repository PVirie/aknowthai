import tensorflow as tf
import numpy as np


def final_layer(lstm_output, w, batches, input_size, output_size):
    """ (batch, data_length, input_size) * (input_size, [one-hots, alpha]) -> (batch, data_length, [one-hots, alpha]) """
    return tf.reshape(tf.matmul(tf.reshape(lstm_output, [-1, input_size]), w), [batches, -1, output_size])


def split_output(final_output, batches, total_classes):
    """ (batch, data_length, [one-hots, alpha]) ->  (batch, data_length, 1), (batch, data_length, one-hots)"""
    alphas = tf.slice(final_output, [0, 0, total_classes], [batches, -1, 1])
    logits = tf.slice(final_output, [0, 0, 0], [batches, -1, total_classes])
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
        self.gpu_inputs = tf.placeholder(tf.float32)
        self.gpu_labels = tf.placeholder(tf.int32)
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

        output, state = tf.nn.dynamic_rnn(self.stacked_lstm, self.gpu_inputs, dtype=tf.float32, time_major=False)
        preAlphas, logits = split_output(final_layer(output, W, batches, input_size, total_classes + 1), batches, total_classes)

        self.alphas = make_prob(tf.mul(preAlphas, self.sharpeness))
        self.classes = logit_to_label(logits)

        sum_cost = tf.Variable(0, trainable=False, dtype=tf.float32)
        with tf.variable_scope("lstm") as scope:
            for i in range(input_sequence_length):
                if i > 0:
                    scope.reuse_variables()
                logits_i = tf.reshape(tf.slice(logits, [0, i, 0], [batches, 1, total_classes]), [batches, total_classes])
                alphas_i = tf.reshape(tf.slice(self.alphas, [0, i, 0], [batches, 1, 1]), [batches, 1])
                shift_mat = build_shifting_graph(alphas_i, tf.constant(np.identity(total_characters), dtype=tf.float32), tf.constant(shift, dtype=tf.float32))
                weights, cost = build_step_cost_function(1.0, shift_mat, alphas_i, weights, logits_i, self.gpu_labels, batches, total_characters, total_classes)
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
            _, loss = self.sess.run((self.training_op, self.overall_cost), feed_dict={self.gpu_inputs: data, self.gpu_labels: labels, self.sharpeness: [step * 100.0 / max_iteration]})
            print loss
        self.saver.save(self.sess, "../artifacts/" + session_name)

    def load_session(self, session_name):
        self.saver.restore(self.sess, "../artifacts/" + session_name)

    def eval(self, data, labels):
        classes, alphas = self.sess.run((self.classes, self.alphas), feed_dict={self.gpu_inputs: data, self.sharpeness: [100.0]})
        # now get only classess corresponding to high alphas
        index_output = np.argmax(classes, axis=2)
        masked = np.reshape(alphas > 0.5, (alphas.shape[0], alphas.shape[1]))

        count = 0
        correct = 0
        for b in xrange(index_output.shape[0]):
            lc = 0
            for c in xrange(index_output.shape[1]):
                if(masked[b, c]):
                    correct += 1 if labels[b, lc] == index_output[b, c] else 0
                    lc += 1
                    count += 1

        print "Percent correct = ", correct * 100.0 / count
