import tensorflow as tf

class NTMCell(tf.contrib.rnn.RNNCell):
    def __init__(self,
                memory, memory_vector_dim, batch_size, num_units, num_layers, source_sequence_length, read_head_num,
                addressing_mode='content', shift_range=2, reuse=False, output_dim=None,
                controller_dropout=0.0, controller_forget_bias=1.0, mode=None, beam_width=None,
                weight_scores=False):
        self.weight_comp = 'luong'
        self.source_sequence_length = source_sequence_length
        self.batch_size = batch_size
        self.memory_vector_dim = memory_vector_dim
        # need to reshape memory in order to get beam search working
        self.memory = tf.reshape(memory, [-1, self.source_sequence_length * self.memory_vector_dim])
        self.num_units = num_units
        self.num_layers = num_layers
        self.read_head_num = read_head_num
        self.addressing_mode = addressing_mode
        self.reuse = reuse

        self.dropout = controller_dropout if mode == tf.contrib.learn.ModeKeys.TRAIN else 0.0
        def single_cell(num_units):
            single_cell = tf.contrib.rnn.BasicLSTMCell(
                num_units,
                forget_bias=controller_forget_bias)
            if self.dropout > 0.0:
              single_cell = tf.contrib.rnn.DropoutWrapper(
                  cell=single_cell, input_keep_prob=(1.0 - self.dropout))
            return single_cell

        self.controller = tf.contrib.rnn.MultiRNNCell([single_cell(self.num_units) for _ in range(num_layers)])
        
        self.step = 0
        self.output_dim = output_dim
        self.shift_range = shift_range
        self.mode = mode
        self.beam_width = beam_width
        self.weight_scores = weight_scores

    def __call__(self, x, prev_state):
        prev_controller_state = prev_state[0]
        prev_read_vector_list = prev_state[1]
        memory = prev_state[3]
        prev_output = prev_state[4]
        prev_output_1 = prev_state[5]
        if self.weight_scores:
            prev_weight_sum = prev_state[-1]
        memory = tf.reshape(memory, [-1, self.source_sequence_length, self.memory_vector_dim])

        controller_input = tf.concat([x] + prev_read_vector_list + [prev_output, prev_output_1], axis=1)
        with tf.variable_scope('controller', reuse=self.reuse):
            controller_output, controller_state = self.controller.__call__(controller_input, prev_controller_state)

        num_parameters_per_head = self.memory_vector_dim + 1 + 1 + (self.shift_range * 2 + 1) + 1
        total_parameter_num = num_parameters_per_head * self.read_head_num
        with tf.variable_scope("o2p", reuse=(self.step > 0) or self.reuse):
            parameters = tf.contrib.layers.fully_connected(controller_output, total_parameter_num,
                activation_fn=None,
                weights_initializer=tf.random_uniform_initializer(-0.1, 0.1))

        head_parameter_list = tf.split(parameters[:, :num_parameters_per_head * self.read_head_num], self.read_head_num, axis=1)

        prev_w_list = prev_state[2]
        read_w_list = []
        for i, head_parameter in enumerate(head_parameter_list):
            # k = tf.tanh(head_parameter[:, 0:self.memory_vector_dim])
            k = head_parameter[:, 0:self.memory_vector_dim]
            beta = tf.nn.softplus(head_parameter[:, self.memory_vector_dim])
            g = tf.sigmoid(head_parameter[:, self.memory_vector_dim + 1])
            s = tf.nn.softmax(
                head_parameter[:, self.memory_vector_dim + 2:self.memory_vector_dim + 2 + (self.shift_range * 2 + 1)]
            )
            gamma = tf.nn.softplus(head_parameter[:, self.memory_vector_dim + 2 + (self.shift_range * 2 + 1)]) + 1
            if self.weight_scores:
                alpha = head_parameter[:, -1]
            with tf.variable_scope('addressing_head_%d' % i):
                # w = self.addressing(k, beta, g, s, gamma, prev_w_list[i], memory,
                #     weight_sum=(1.0/(tf.expand_dims(alpha, axis=1) * prev_weight_sum[i])) if self.weight_scores else None)
                w = self.addressing(k, beta, g, s, gamma, prev_w_list[i], memory,
                    weight_sum=(1.0/(tf.expand_dims(alpha, axis=1) * tf.exp(prev_weight_sum[i]))) if self.weight_scores else None)
            read_w_list.append(w)

        read_vector_list = []
        weight_sum = []
        for i in range(self.read_head_num):
            if self.weight_scores:
                weight_sum.append(prev_weight_sum[i] + read_w_list[i])
            read_vector = tf.reduce_sum(tf.expand_dims(read_w_list[i], dim=2) * memory, axis=1)
            read_vector_list.append(read_vector)

        if not self.output_dim:
            self.output_dim = x.get_shape()[1]
        with tf.variable_scope("o2o", reuse=(self.step > 0) or self.reuse):
            # controller_output = tf.concat([controller_output] + read_vector_list, axis=1)
            NTM_output = tf.contrib.layers.fully_connected(controller_output, self.output_dim,
                activation_fn=tf.tanh,
                weights_initializer=tf.random_uniform_initializer(-0.1, 0.1))

        memory = tf.reshape(memory, [-1, self.source_sequence_length * self.memory_vector_dim])

        self.step += 1

        output = tf.contrib.layers.fully_connected(tf.concat([NTM_output, tf.concat(read_vector_list, axis=1)], axis=1),
            self.output_dim,
            activation_fn=tf.tanh,
            weights_initializer=tf.random_uniform_initializer(-0.1, 0.1))

        if self.dropout > 0.0:
            output = tf.nn.dropout(output, 1-self.dropout)

        if self.weight_scores:
            return output, (controller_state, read_vector_list, read_w_list, memory, output, prev_output, weight_sum)
        else:
            return output, (controller_state, read_vector_list, read_w_list, memory, output, prev_output)

    def addressing(self, k, beta, g, s, gamma, prev_w, memory, weight_sum=None):

        # Sec 3.3.1 Focusing by Content

        # Cosine Similarity

        k = tf.expand_dims(k, axis=2)
        inner_product = tf.matmul(memory, k)

        if weight_sum is not None:
            inner_product = inner_product * tf.expand_dims(weight_sum, axis=2)

        if self.weight_comp == 'luong':                                   # Only focus on content
            inner_product = tf.squeeze(inner_product, axis=2)
            w_c = tf.nn.softmax(tf.expand_dims(beta, axis=1) * inner_product, dim=1)
        else:
            # k_norm = tf.sqrt(tf.reduce_sum(tf.square(k), axis=1, keep_dims=True))
            # M_norm = tf.sqrt(tf.reduce_sum(tf.square(memory), axis=2, keep_dims=True))

            k_norm = tf.norm(k, axis=1)
            M_norm = tf.norm(memory, axis=2)
            norm_product = tf.expand_dims(M_norm * k_norm, axis=2)

            cosine_sim = tf.squeeze(inner_product / (norm_product + 1e-8))
            cosine_sim.set_shape([None, self.source_sequence_length])

            w_c = tf.nn.softmax(tf.expand_dims(beta, axis=1) * cosine_sim, dim=1)

            # K = tf.squeeze(inner_product / (norm_product + 1e-8))                   # eq (6)

            # # Calculating w^c

            # K_amplified = tf.exp(tf.expand_dims(beta, axis=1) * K)
            # w_c = K_amplified / tf.reduce_sum(K_amplified, axis=1, keep_dims=True)  # eq (5)

            w_c.set_shape([None, self.source_sequence_length])

        if self.addressing_mode == 'content':                                   # Only focus on content
            return w_c

        if self.addressing_mode == 'sharpening_only':
            w_sharpen = tf.pow(w_c, tf.expand_dims(gamma, axis=1))
            return w_sharpen / tf.reduce_sum(w_sharpen, axis=1, keep_dims=True)

        # Sec 3.3.2 Focusing by Location

        g = tf.expand_dims(g, axis=1)
        w_g = g * w_c + (1 - g) * prev_w                                        # eq (7)

        if self.addressing_mode == 'content_and_interpolation':                                   # Only focus on content
            return w_g

        s = tf.concat([s[:, :self.shift_range + 1],
                       tf.zeros([self.batch_size, self.source_sequence_length - (self.shift_range * 2 + 1)]),
                       s[:, -self.shift_range:]], axis=1)
        t = tf.concat([tf.reverse(s, axis=[1]), tf.reverse(s, axis=[1])], axis=1)
        s_matrix = tf.stack(
            [t[:, self.source_sequence_length - i - 1:self.source_sequence_length * 2 - i - 1] for i in range(self.source_sequence_length)],
            axis=1
        )
        if self.addressing_mode == 'content_and_shift_no_interp':
            w_ = tf.reduce_sum(tf.expand_dims(w_c, axis=1) * s_matrix, axis=2)
        else:
            w_ = tf.reduce_sum(tf.expand_dims(w_g, axis=1) * s_matrix, axis=2)      # eq (8)

        if self.addressing_mode == 'content_and_shift':                                   # Only focus on content
            return w_

        w_sharpen = tf.pow(w_, tf.expand_dims(gamma, axis=1))
        w = w_sharpen / tf.reduce_sum(w_sharpen, axis=1, keep_dims=True)        # eq (9)

        return w

    def zero_state(self, batch_size, dtype):
        def expand(x, dim, N):
            return tf.concat([tf.expand_dims(x, dim) for _ in tf.range(N)], axis=dim)

        with tf.variable_scope('init', reuse=self.reuse):
            controller_state = self.controller.zero_state(batch_size, dtype)
            read_vector_list = [tf.zeros([batch_size, self.memory_vector_dim])
                for _ in range(self.read_head_num)]
            read_w_list = [tf.zeros([batch_size, self.source_sequence_length])
                for _ in range(self.read_head_num)]
            memory = self.memory
            output = tf.zeros([batch_size, self.output_dim])
            output_1 = tf.zeros([batch_size, self.output_dim])

            if self.weight_scores:
                weight_sum = [tf.zeros([batch_size, self.source_sequence_length])
                    for _ in range(self.read_head_num)]

                return (controller_state, read_vector_list, read_w_list, memory, output, output_1, weight_sum)
            else:
                return (controller_state, read_vector_list, read_w_list, memory, output, output_1)
    
    @property
    def state_size(self):
        if self.weight_scores:
            return (self.controller.state_size,
                [self.memory_vector_dim for _ in range(self.read_head_num)],
                [self.source_sequence_length for _ in range(self.read_head_num)],
                tf.TensorShape([self.source_sequence_length * self.memory_vector_dim]),
                tf.TensorShape([self.output_dim]),
                tf.TensorShape([self.output_dim]),
                [self.source_sequence_length for _ in range(self.read_head_num)])
        else:
            return (self.controller.state_size,
                [self.memory_vector_dim for _ in range(self.read_head_num)],
                [self.source_sequence_length for _ in range(self.read_head_num)],
                tf.TensorShape([self.source_sequence_length * self.memory_vector_dim]),
                tf.TensorShape([self.output_dim]),
                tf.TensorShape([self.output_dim]))

    @property
    def output_size(self):
        return self.controller.output_size

