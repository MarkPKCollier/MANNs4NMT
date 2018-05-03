# credit: thanks to https://github.com/snowkylin/ntm for the basis of this code
# the major changes made are to make this compatible with the abstract class tf.contrib.rnn.RNNCell
# additionally an LSTM controller is used, a feed-forward controller may be used
# and 2 memory inititialization schemes are offered

import tensorflow as tf
from collections import namedtuple
from ntm_utils import expand, learned_init, create_linear_initializer

Model1NTMState = namedtuple('Model1NTMState',
        ('controller_state', 'time', 'prev_output',
        'att_read_vector_list', 'att_w_list', 'att_w_history', 'att_M'))

Model2NTMState = namedtuple('Model2NTMState',
        ('controller_state', 'time', 'prev_output',
        'ext_read_vector_list', 'ext_w_list', 'ext_w_history', 'ext_M',
        'att_read_vector_list', 'att_w_list', 'att_w_history', 'att_M'))

Model3NTMState = namedtuple('Model3NTMState',
        ('controller_state', 'time', 'prev_output',
        'ext_read_vector_list', 'ext_w_list', 'ext_w_history', 'ext_M'))

class NTMCell(tf.contrib.rnn.RNNCell):
    def __init__(self, controller_layers, controller_units,
                use_att_memory=True, att_memory=None, att_memory_size=None, att_memory_vector_dim=None,
                use_ext_memory=False, ext_memory_size=None, ext_memory_vector_dim=None, ext_read_head_num=None, ext_write_head_num=None,
                dropout=0.0, batch_size=None, mode=None, addressing_mode='content_and_location',
                shift_range=1, reuse=False, output_dim=None, clip_value=20, record_w_history=False):
        self.controller_layers = controller_layers
        self.controller_units = controller_units

        self.att_memory_size = att_memory_size
        self.att_memory_vector_dim = att_memory_vector_dim
        self.ext_memory_size = ext_memory_size
        self.ext_memory_vector_dim = ext_memory_vector_dim

        self.att_read_head_num = 1 if att_memory is not None else 0
        self.ext_read_head_num = ext_read_head_num
        self.total_read_head_num = self.att_read_head_num + (ext_read_head_num if ext_read_head_num is not None else 0)
        self.ext_write_head_num = ext_write_head_num

        self.use_att_memory = use_att_memory
        self.use_ext_memory = use_ext_memory

        # need to reshape memory in order to get beam search working
        if self.use_att_memory:
            self.att_M = tf.reshape(att_memory, [-1, self.att_memory_size * self.att_memory_vector_dim])
        else:
            self.att_M = None

        self.addressing_mode = addressing_mode
        self.reuse = reuse
        self.clip_value = clip_value

        self.dropout = dropout if mode == tf.contrib.learn.ModeKeys.TRAIN else 0.0
        def single_cell(num_units):
            cell = tf.contrib.rnn.BasicLSTMCell(num_units, forget_bias=1.0)
            if self.dropout > 0.0:
                cell = tf.contrib.rnn.DropoutWrapper(cell=cell, input_keep_prob=(1.0 - self.dropout))
            return cell

        self.controller = tf.contrib.rnn.MultiRNNCell([single_cell(self.controller_units) for _ in range(self.controller_layers)])

        self.step = 0
        self.output_dim = output_dim
        self.shift_range = shift_range
        self.batch_size = batch_size
        self.mode = mode
        self.record_w_history = record_w_history

        self.o2p_initializer = create_linear_initializer(self.controller_units)
        self.o2o_initializer = create_linear_initializer(
            self.controller_units + \
            (self.att_memory_vector_dim if self.use_att_memory else 0) + \
            (self.ext_memory_vector_dim * self.ext_read_head_num if self.use_ext_memory else 0))

    def interact_with_memory(self, prev_state, controller_output, att=True):
        num_parameters_per_head = (self.att_memory_vector_dim if att else self.ext_memory_vector_dim) + 1 + 1 + (self.shift_range * 2 + 1) + 1
        num_heads = 1 if att else (self.ext_read_head_num + self.ext_write_head_num)
        total_parameter_num = num_parameters_per_head if att else (num_parameters_per_head * num_heads + self.ext_memory_vector_dim * 2 * self.ext_write_head_num)
        with tf.variable_scope("o2p_att_" + str(att), reuse=tf.AUTO_REUSE):
            parameters = tf.contrib.layers.fully_connected(
                controller_output, total_parameter_num, activation_fn=None,
                weights_initializer=self.o2p_initializer)
            parameters = tf.clip_by_value(parameters, -self.clip_value, self.clip_value)
        head_parameter_list = tf.split(parameters[:, :num_parameters_per_head * num_heads], num_heads, axis=1)

        if att:
            prev_w_list = prev_state.att_w_list
            prev_M = prev_state.att_M
            prev_M = tf.reshape(prev_M, [-1, self.att_memory_size, self.att_memory_vector_dim])
            memory_vector_dim = self.att_memory_vector_dim
        else:
            prev_w_list = prev_state.ext_w_list
            prev_M = prev_state.ext_M
            prev_M = tf.reshape(prev_M, [-1, self.ext_memory_size, self.ext_memory_vector_dim])
            memory_vector_dim = self.ext_memory_vector_dim
        
        w_list = []
        for i, head_parameter in enumerate(head_parameter_list):
            k = tf.tanh(head_parameter[:, 0:memory_vector_dim])
            beta = tf.nn.softplus(head_parameter[:, memory_vector_dim])
            g = tf.sigmoid(head_parameter[:, memory_vector_dim + 1])
            s = tf.nn.softmax(
                head_parameter[:, memory_vector_dim + 2:memory_vector_dim + 2 + (self.shift_range * 2 + 1)]
            )
            gamma = tf.nn.softplus(head_parameter[:, -1]) + 1
            with tf.variable_scope('addressing_head_%d' % i):
                w = self.addressing(k, beta, g, s, gamma, prev_M, prev_w_list[i], att=att)
            w_list.append(w)

        # Reading (Sec 3.1)

        if att:
            read_vector_list = [tf.reduce_sum(tf.expand_dims(w_list[0], dim=2) * prev_M, axis=1)]
        else:
            read_w_list = w_list[:self.ext_read_head_num]
            read_vector_list = []
            for i in range(self.ext_read_head_num):
                read_vector = tf.reduce_sum(tf.expand_dims(read_w_list[i], dim=2) * prev_M, axis=1)
                read_vector_list.append(read_vector)

        # Writing (Sec 3.2)

        M = prev_M
        if not att:
            erase_add_list = tf.split(parameters[:, num_parameters_per_head * num_heads:], 2 * self.ext_write_head_num, axis=1)
            write_w_list = w_list[self.ext_read_head_num:]
            for i in range(self.ext_write_head_num):
                w = tf.expand_dims(write_w_list[i], axis=2)
                erase_vector = tf.expand_dims(tf.sigmoid(erase_add_list[i * 2]), axis=1)
                add_vector = tf.expand_dims(tf.tanh(erase_add_list[i * 2 + 1]), axis=1)
                M = M * (tf.ones([self.batch_size, self.ext_memory_size, self.ext_memory_vector_dim]) - tf.matmul(w, erase_vector)) + tf.matmul(w, add_vector)

        return read_vector_list, w_list, M

    def __call__(self, x, prev_state):
        if self.use_att_memory and self.use_ext_memory:
            prev_state = Model2NTMState(*prev_state)
        elif self.use_att_memory:
            prev_state = Model1NTMState(*prev_state)
        else:
            prev_state = Model3NTMState(*prev_state)

        prev_read_vector_list = (prev_state.ext_read_vector_list if self.use_ext_memory else []) + \
            (prev_state.att_read_vector_list if self.use_att_memory else [])

        controller_input = tf.concat([x] + prev_read_vector_list + [prev_state.prev_output], axis=1)
        with tf.variable_scope('controller', reuse=self.reuse):
            controller_output, controller_state = self.controller(controller_input, prev_state.controller_state)

        if self.use_att_memory:
            att_read_vector_list, att_w_list, att_M = self.interact_with_memory(prev_state, controller_output, att=True)
            att_M = tf.reshape(att_M, [-1, self.att_memory_size * self.att_memory_vector_dim])

        if self.use_ext_memory:
            ext_read_vector_list, ext_w_list, ext_M = self.interact_with_memory(prev_state, controller_output, att=False)
            ext_M = tf.reshape(ext_M, [-1, self.ext_memory_size * self.ext_memory_vector_dim])

        if not self.output_dim:
            output_dim = x.get_shape()[1]
        else:
            output_dim = self.output_dim
        with tf.variable_scope("o2o", reuse=tf.AUTO_REUSE):
            read_vector_list = (ext_read_vector_list if self.use_ext_memory else []) + \
                (att_read_vector_list if self.use_att_memory else [])

            NTM_output = tf.contrib.layers.fully_connected(
                tf.concat([controller_output] + read_vector_list, axis=1), output_dim, activation_fn=None,
                weights_initializer=self.o2o_initializer)
            NTM_output = tf.clip_by_value(NTM_output, -self.clip_value, self.clip_value)

        if self.dropout > 0.0:
            NTM_output = tf.nn.dropout(NTM_output, 1-self.dropout)

        self.step += 1

        if self.use_att_memory:
            map(lambda v: v.set_shape([None, self.att_memory_vector_dim]), att_read_vector_list)
            map(lambda v: v.set_shape([None, self.att_memory_size]), att_w_list)
        if self.use_ext_memory:
            map(lambda v: v.set_shape([None, self.ext_memory_vector_dim]), ext_read_vector_list)
            map(lambda v: v.set_shape([None, self.ext_memory_size]), ext_w_list)

        if self.use_att_memory and self.use_ext_memory:
            return NTM_output, tuple(Model2NTMState(
                time=prev_state.time + 1 if self.record_w_history else prev_state.time,
                controller_state=controller_state,
                ext_read_vector_list=ext_read_vector_list,
                ext_w_list=ext_w_list,
                ext_w_history=[prev_state.ext_w_history[i].write(prev_state.time, ext_w_list[i]) for i in range(self.ext_read_head_num + self.ext_write_head_num)] if self.record_w_history else prev_state.ext_w_history,
                ext_M=ext_M,
                att_read_vector_list=att_read_vector_list,
                att_w_list=att_w_list,
                att_w_history=prev_state.att_w_history.write(prev_state.time, att_w_list[0]) if self.record_w_history else prev_state.att_w_history,
                att_M=att_M,
                prev_output=NTM_output))
        elif self.use_att_memory:
            return NTM_output, tuple(Model1NTMState(
                time=prev_state.time + 1 if self.record_w_history else prev_state.time,
                controller_state=controller_state,
                att_read_vector_list=att_read_vector_list,
                att_w_list=att_w_list,
                att_w_history=prev_state.att_w_history.write(prev_state.time, att_w_list[0]) if self.record_w_history else prev_state.att_w_history,
                att_M=att_M,
                prev_output=NTM_output))
        else:
            return NTM_output, tuple(Model3NTMState(
                time=prev_state.time + 1 if self.record_w_history else prev_state.time,
                controller_state=controller_state,
                ext_read_vector_list=ext_read_vector_list,
                ext_w_list=ext_w_list,
                ext_w_history=[prev_state.ext_w_history[i].write(prev_state.time, ext_w_list[i]) for i in range(self.ext_read_head_num + self.ext_write_head_num)] if self.record_w_history else prev_state.ext_w_history,
                ext_M=ext_M,
                prev_output=NTM_output))

    def addressing(self, k, beta, g, s, gamma, prev_M, prev_w, att=True):
        k = tf.expand_dims(k, axis=2)
        inner_product = tf.matmul(prev_M, k)
        if att:
            inner_product = tf.squeeze(inner_product, axis=2)
            w_c = tf.nn.softmax(tf.expand_dims(beta, axis=1) * inner_product, dim=1)
        else:
            k_norm = tf.sqrt(tf.reduce_sum(tf.square(k), axis=1, keep_dims=True))
            M_norm = tf.sqrt(tf.reduce_sum(tf.square(prev_M), axis=2, keep_dims=True))
            norm_product = M_norm * k_norm
            K = tf.squeeze(inner_product / (norm_product + 1e-8))                   # eq (6)

            # Calculating w^c

            K_amplified = tf.exp(tf.expand_dims(beta, axis=1) * K)
            w_c = K_amplified / tf.reduce_sum(K_amplified, axis=1, keep_dims=True)  # eq (5)

            w_c = tf.squeeze(w_c)

        if att:
            w_c.set_shape([None, self.att_memory_size])
        else:
            w_c.set_shape([None, self.ext_memory_size])

        if self.addressing_mode == 'content':                                   # Only focus on content
            return w_c

        # Sec 3.3.2 Focusing by Location

        g = tf.expand_dims(g, axis=1)
        w_g = g * w_c + (1 - g) * prev_w                                        # eq (7)

        if att:
            s = tf.concat([s[:, :self.shift_range + 1],
                           tf.zeros([self.batch_size, self.att_memory_size - (self.shift_range * 2 + 1)]),
                           s[:, -self.shift_range:]], axis=1)
            t = tf.concat([tf.reverse(s, axis=[1]), tf.reverse(s, axis=[1])], axis=1)
            s_matrix = tf.stack(
                [t[:, self.att_memory_size - i - 1:self.att_memory_size * 2 - i - 1] for i in range(self.att_memory_size)],
                axis=1
            )
        else:
            s = tf.concat([s[:, :self.shift_range + 1],
                           tf.zeros([self.batch_size, self.ext_memory_size - (self.shift_range * 2 + 1)]),
                           s[:, -self.shift_range:]], axis=1)
            t = tf.concat([tf.reverse(s, axis=[1]), tf.reverse(s, axis=[1])], axis=1)
            s_matrix = tf.stack(
                [t[:, self.ext_memory_size - i - 1:self.ext_memory_size * 2 - i - 1] for i in range(self.ext_memory_size)],
                axis=1
            )
        
        w_ = tf.reduce_sum(tf.expand_dims(w_g, axis=1) * s_matrix, axis=2)      # eq (8)
        w_sharpen = tf.pow(w_, tf.expand_dims(gamma, axis=1))
        w = w_sharpen / tf.reduce_sum(w_sharpen, axis=1, keep_dims=True)        # eq (9)

        return w

    def zero_state(self, batch_size, dtype):
        with tf.variable_scope('init', reuse=self.reuse):
            controller_init_state = self.controller.zero_state(batch_size, dtype)
            prev_output = tf.zeros([batch_size, self.output_dim])

            if self.use_ext_memory:
                ext_read_vector_list = [expand(tf.tanh(learned_init(self.ext_memory_vector_dim)), dim=0, N=batch_size, dims=1)
                    for i in range(self.ext_read_head_num)]

                ext_w_list = [expand(tf.nn.softmax(learned_init(self.ext_memory_size)), dim=0, N=batch_size, dims=1)
                    for i in range(self.ext_read_head_num + self.ext_write_head_num)]

                # ext_M = expand(tf.tanh(learned_init(self.ext_memory_size * self.ext_memory_vector_dim)), dim=0, N=batch_size, dims=1)

                ext_M = expand(tf.get_variable('init_M', self.ext_memory_size * self.ext_memory_vector_dim,
                    initializer=tf.constant_initializer(1e-6)),
                    dim=0, N=batch_size, dims=1)

            if self.use_att_memory:
                att_read_vector_list = [expand(tf.tanh(learned_init(self.att_memory_vector_dim)), dim=0, N=batch_size, dims=1)]
                att_w_list = [expand(tf.nn.softmax(learned_init(self.att_memory_size)), dim=0, N=batch_size, dims=1)]

            if self.use_att_memory and self.use_ext_memory:
                # tmp_att_M = tf.reshape(self.att_M, [-1, self.att_memory_size, self.att_memory_vector_dim])
                # m = tf.contrib.layers.fully_connected(tf.reduce_mean(tmp_att_M, axis=1), self.ext_memory_vector_dim,
                #     activation_fn=tf.tanh, weights_initializer=create_linear_initializer(self.att_memory_vector_dim))
                # ext_M = tf.tile(tf.expand_dims(m, 1), multiples=[1, self.ext_memory_size, 1]) + tf.random_normal([batch_size, self.ext_memory_size, self.ext_memory_vector_dim], stddev=0.316)
                # ext_M = tf.reshape(ext_M, [-1, self.ext_memory_size * self.ext_memory_vector_dim])

                # return tuple(Model2NTMState(
                #     controller_state=controller_init_state,
                #     ext_read_vector_list=ext_read_vector_list,
                #     ext_w_list=ext_w_list,
                #     ext_M=ext_M,
                #     att_read_vector_list=att_read_vector_list,
                #     att_w_list=att_w_list,
                #     att_M=self.att_M,
                #     prev_output=prev_output))

                return tuple(Model2NTMState(
                    time=tf.zeros([], dtype=tf.int32) if self.record_w_history else tf.zeros([batch_size, 1], dtype=tf.int32),
                    controller_state=controller_init_state,
                    ext_read_vector_list=ext_read_vector_list,
                    ext_w_list=ext_w_list,
                    ext_w_history=[tf.TensorArray(dtype=dtype, size=0, dynamic_size=True) for _ in range(self.ext_read_head_num + self.ext_write_head_num)] if self.record_w_history else tf.zeros([batch_size, 1], dtype=tf.int32),
                    ext_M=ext_M,
                    att_read_vector_list=att_read_vector_list,
                    att_w_list=att_w_list,
                    att_w_history=tf.TensorArray(dtype=dtype, size=0, dynamic_size=True) if self.record_w_history else tf.zeros([batch_size, 1], dtype=tf.int32),
                    att_M=self.att_M,
                    prev_output=prev_output))
            elif self.use_att_memory:
                return tuple(Model1NTMState(
                    time=tf.zeros([], dtype=tf.int32) if self.record_w_history else tf.zeros([batch_size, 1], dtype=tf.int32),
                    controller_state=controller_init_state,
                    att_read_vector_list=att_read_vector_list,
                    att_w_list=att_w_list,
                    att_w_history=tf.TensorArray(dtype=dtype, size=0, dynamic_size=True)if self.record_w_history else tf.zeros([batch_size, 1], dtype=tf.int32),
                    att_M=self.att_M,
                    prev_output=prev_output))
            else:
                return tuple(Model3NTMState(
                    time=tf.zeros([], dtype=tf.int32) if self.record_w_history else tf.zeros([batch_size, 1], dtype=tf.int32),
                    controller_state=controller_init_state,
                    ext_read_vector_list=ext_read_vector_list,
                    ext_w_list=ext_w_list,
                    ext_w_history=[tf.TensorArray(dtype=dtype, size=0, dynamic_size=True) for _ in range(self.ext_read_head_num + self.ext_write_head_num)] if self.record_w_history else tf.zeros([batch_size, 1], dtype=tf.int32),
                    ext_M=ext_M,
                    prev_output=prev_output))

    @property
    def state_size(self):
        if self.use_att_memory and self.use_ext_memory:
            return tuple(Model2NTMState(
                time=tf.TensorShape([]) if self.record_w_history else tf.TensorShape([1]),
                controller_state=self.controller.state_size,
                ext_read_vector_list=[self.ext_memory_vector_dim for _ in range(self.ext_read_head_num)],
                ext_w_list=[self.ext_memory_size for _ in range(self.ext_read_head_num + self.ext_write_head_num)],
                ext_w_history=[tuple() for _ in range(self.ext_read_head_num + self.ext_write_head_num)] if self.record_w_history else tf.TensorShape([1]),
                ext_M=tf.TensorShape([self.ext_memory_size * self.ext_memory_vector_dim]),
                att_read_vector_list=[self.att_memory_vector_dim],
                att_w_list=[self.att_memory_size],
                att_w_history=tuple() if self.record_w_history else tf.TensorShape([1]),
                att_M=tf.TensorShape([self.att_memory_size * self.att_memory_vector_dim]),
                prev_output=tf.TensorShape([self.output_dim])))
        elif self.use_att_memory:
            return tuple(Model1NTMState(
                time=tf.TensorShape([]) if self.record_w_history else tf.TensorShape([1]),
                controller_state=self.controller.state_size,
                att_read_vector_list=[self.att_memory_vector_dim],
                att_w_list=[self.att_memory_size],
                att_w_history=tuple() if self.record_w_history else tf.TensorShape([1]),
                att_M=tf.TensorShape([self.att_memory_size * self.att_memory_vector_dim]),
                prev_output=tf.TensorShape([self.output_dim])))
        else:
            return tuple(Model3NTMState(
                time=tf.TensorShape([]) if self.record_w_history else tf.TensorShape([1]),
                controller_state=self.controller.state_size,
                ext_read_vector_list=[self.ext_memory_vector_dim for _ in range(self.ext_read_head_num)],
                ext_w_list=[self.ext_memory_size for _ in range(self.ext_read_head_num + self.ext_write_head_num)],
                ext_w_history=[tuple() for _ in range(self.ext_read_head_num + self.ext_write_head_num)] if self.record_w_history else tf.TensorShape([1]),
                ext_M=tf.TensorShape([self.ext_memory_size * self.ext_memory_vector_dim]),
                prev_output=tf.TensorShape([self.output_dim])))

    @property
    def output_size(self):
        return self.output_dim
