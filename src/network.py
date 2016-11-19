import tensorflow as tf
import numpy as np


def linear_layer(x, w, b, input_size, output_size):
    """ (batch, data_length, input_size) * (input_size, output_size) -> (batch, data_length, output_size) """
    batches = tf.shape(x)[0]
    return tf.reshape(tf.matmul(tf.reshape(x, [-1, input_size]), w), [batches, -1, output_size]) + b


def split_output(final_output, total_classes):
    """ (batch, data_length, [one-hots, alpha]) ->  (batch, data_length, 1), (batch, data_length, one-hots)"""
    batches = tf.shape(final_output)[0]
    alphas = tf.slice(final_output, [0, 0, 0], [batches, -1, 1])
    logits = tf.slice(final_output, [0, 0, 1], [batches, -1, total_classes])
    return alphas, logits


def logit_to_label(logits):
    """ activated labels must be forced sparse! """
    return tf.nn.softmax(logits)


def make_prob(preAlphas):
    """ preAlphas will be normalized into range [0, 1] """
    return tf.sigmoid(preAlphas)


def build_shifting_graph(alphas, identity, shift_template):
    """ for both tensorflow and numpy tensors """
    return identity + (alphas * shift_template)


def build_step_output(shift, alphas, weights, classes):
    """ total_characters = true_character_number + 1, the last slot is for terminal penalty """
    """ true_labels of integer type can be expaned into [batches, total_characters, total_classes] via tf.one_hot """
    """ classes [batches, 1, total_classes] """
    """ weights [batches, total_characters, 1] """
    """ alphas [batches, 1, 1]  """
    """ shift [batches, total_characters, total_characters] """
    out = tf.mul(alphas, tf.mul(weights, classes))
    new_weights = tf.batch_matmul(shift, weights)
    return new_weights, out


def build_shift_identiy_weights(start_weight, total_batches, total_characters, total_classes):
    r12340 = np.arange(1, total_characters + 1, 1, dtype=np.int32)
    r12340[total_characters - 1] = 0
    cpu_shift = -np.identity(total_characters) + np.identity(total_characters)[:, r12340]
    cpu_shift[0, total_characters - 1] = 0
    cpu_shift[total_characters - 1, total_characters - 1] = 0

    cpu_identity = np.identity(total_characters)

    cpu_weights = np.ones((total_batches, total_characters, 1)) * (1.0 - start_weight) / total_characters
    cpu_weights[:, 0] = start_weight

    return cpu_shift, cpu_identity, cpu_weights


def gen_output(alphas, classes, start_weight, total_batches, total_characters, total_classes, input_sequence_length):

    cpu_shift, cpu_identity, cpu_weights = build_shift_identiy_weights(start_weight, total_batches, total_characters, total_classes)
    shift = tf.constant(cpu_shift, dtype=tf.float32)
    identity = tf.constant(cpu_identity, dtype=tf.float32)
    weights = tf.constant(cpu_weights, dtype=tf.float32)

    z = tf.zeros([total_batches, total_characters, total_classes], dtype=tf.float32)
    for i in range(input_sequence_length):
        classes_i = tf.slice(classes, [0, i, 0], [-1, 1, total_classes])
        alphas_i = tf.slice(alphas, [0, i, 0], [-1, 1, 1])
        shift_mat = build_shifting_graph(alphas_i, identity, shift)
        weights, out = build_step_output(shift_mat, alphas_i, weights, classes_i)
        z = z + out
    return z


def gen_cpu_output(alphas, classes, start_weight, total_batches, total_classes, input_sequence_length):

    cpu_shift, cpu_identity, weights = build_shift_identiy_weights(start_weight, total_batches, input_sequence_length, total_classes)

    z = np.zeros((total_batches, input_sequence_length, total_classes), dtype=np.float32)
    for i in range(input_sequence_length):
        classes_i = classes[:, i:(i + 1), :]
        alphas_i = alphas[:, i:(i + 1), :]
        shift_mat = build_shifting_graph(alphas_i, cpu_identity, cpu_shift)

        out = alphas_i * (weights * classes_i)
        weights = np.matmul(shift_mat, weights)
        z = z + out
    return z


class Network:

    def __init__(self, data_shape, labels_shape, total_classes, lstm_size):
        self.sess = tf.Session()
        total_batches = data_shape[0]
        total_characters = labels_shape[1]
        input_size = data_shape[2]
        input_sequence_length = data_shape[1]
        self.gpu_inputs = tf.placeholder(tf.float32, [None, None, input_size])
        self.gpu_labels = tf.placeholder(tf.int32)

        with tf.variable_scope("lstm"):
            lstm = tf.nn.rnn_cell.LSTMCell(lstm_size, num_proj=total_classes + 1, forget_bias=0.5)
            self.W = tf.Variable((np.random.rand(input_size, lstm_size) - 0.5) * 0.01, dtype=tf.float32)
            self.b = tf.Variable(np.zeros((lstm_size)), dtype=tf.float32)
            self.stacked_lstm = tf.nn.rnn_cell.MultiRNNCell([lstm] * 2)
            preLSTM = tf.tanh(linear_layer(self.gpu_inputs, self.W, self.b, input_size, lstm_size))
            output, state = tf.nn.dynamic_rnn(self.stacked_lstm, preLSTM, dtype=tf.float32, time_major=False, parallel_iterations=1, swap_memory=True)
            preAlphas, logits = split_output(output, total_classes)
            self.alphas = make_prob(preAlphas)
            self.classes = logit_to_label(logits)

        lstm_scope = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="lstm")

        z = gen_output(self.alphas, self.classes, 0.95, total_batches, total_characters, total_classes, input_sequence_length)
        y = tf.reshape(tf.one_hot(self.gpu_labels, depth=total_classes, dtype=tf.float32, axis=-1), [total_batches, total_characters, total_classes])
        size = [total_batches, total_characters, 1]
        mask = tf.select(tf.reshape(tf.greater(self.gpu_labels, 0), size), tf.ones(size, dtype=tf.float32), tf.zeros(size, dtype=tf.float32))
        self.overall_cost = tf.reduce_sum(tf.mul(mask, -tf.mul(y, tf.log(z))))

        """ out of many algorithms, only Adam converge! A remarkable job for Kingma and Lei Ba!"""
        self.training_op = tf.train.AdamOptimizer(0.001).minimize(self.overall_cost, var_list=lstm_scope)
        self.saver = tf.train.Saver(var_list=lstm_scope, keep_checkpoint_every_n_hours=1)

        # Before starting, initialize the variables.  We will 'run' this first.
        init = tf.initialize_all_variables()

        # Launch the graph.
        self.sess.run(init)

    def train(self, data, labels, session_name, batch_size, max_iteration, continue_from_last=False):
        if continue_from_last:
            self.load_session(session_name)

        for step in xrange(max_iteration):
            sum_loss = 0.0
            total_batches = data.shape[0] / batch_size
            for b in xrange(total_batches):
                db = data[(b * batch_size):((b + 1) * batch_size), ...]
                lb = labels[(b * batch_size):((b + 1) * batch_size), ...]
                _, loss = self.sess.run((self.training_op, self.overall_cost), feed_dict={self.gpu_inputs: db, self.gpu_labels: lb})
                sum_loss += loss
            print sum_loss / total_batches
            if step % 1000 == 0:
                self.saver.save(self.sess, "../artifacts/" + session_name)

        self.saver.save(self.sess, "../artifacts/" + session_name)

    def load_session(self, session_name):
        print "loading from last save..."
        self.saver.restore(self.sess, "../artifacts/" + session_name)

    def load_last(self):
        self.saver.restore(self.sess, tf.train.latest_checkpoint("../artifacts/"))

    def scan(self, data, total_classes):
        classes, alphas = self.sess.run((self.classes, self.alphas), feed_dict={self.gpu_inputs: data})
        out = gen_cpu_output(alphas, classes, 1.0, data.shape[0], total_classes, data.shape[1])
        return out, alphas
