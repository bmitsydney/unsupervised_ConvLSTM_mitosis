import tensorflow as tf
import numpy as np
import os
from copy import deepcopy
from skimage.measure import label
from scipy.ndimage import center_of_mass, label, filters, binary_erosion
import time

class ConvLSTM(object):
    def __init__(self, session,
                 optimizer,
                 saver,
                 checkpoint_dir,
                 max_gradient=5,
                 summary_writer=None,
                 summary_every=100,
                 save_every=2000,
                 training=True,
                 size_x=64,
                 size_y=64,
                 seq_len=10,
                 batch_size=4,
                 training_data=None,
                 training_labels=None,
                 training_frames=None,
                 channels=2,
                 z_margin=4):
        self.session = session
        self.optimizer = optimizer
        self.saver = saver
        self.max_gradient = max_gradient
        self.summary_writer = summary_writer
        self.summary_every = summary_every
        self.save_every = save_every
        self.checkpoint_dir = checkpoint_dir
        self.training = training

        self.batch_size = batch_size
        self.seq_len = seq_len
        self.size_x = training_data.shape[1]
        self.size_y = training_data.shape[2]
        self.training_data = np.expand_dims(training_data, -1)  # np.vstack((training_data, training_data[-1:, :, :]))
        self.training_channels = training_data.shape[-1]
        self.training_labels = np.expand_dims(training_labels, -1)
        self.training_frames = np.expand_dims(training_frames, -1)
        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        self.epoch_size = int(self.training_data.shape[0] / self.batch_size)
        self.z_margin = z_margin

        self.create_variables()
        self.summary_writer.add_graph(self.session.graph)

        self.compress_jump = 2
        self.z_tolerance = 1
        self.xy_tolerance = 10 / self.compress_jump

    def create_variables(self):
        self.input_1 = tf.placeholder(tf.float32, [self.batch_size, self.seq_len, self.size_x, self.size_y, 1], name='input1')
        self.input_2 = tf.placeholder(tf.float32, [self.batch_size, self.seq_len, self.size_x, self.size_y, 1], name='input2')


        self.output = self.model()

        params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

        self.loss = tf.reduce_sum(tf.losses.mean_squared_error(labels=self.input_2, predictions=self.output))

        self.lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in params if 'bias' not in v.name]) * 1e-4
        self.loss += self.lossL2

        self.train_op = self.optimizer.minimize(self.loss, var_list=params, global_step=self.global_step)

        ones = tf.ones_like(self.input_1)
        images = tf.concat([self.input_1[0, :, :, :, :],
                            ones[0, :, :, 0:1, :], self.input_2[0, :, :, :, :],
                            ones[0, :, :, 0:1, :], self.output[0, :, :, :, :]], -2)
        image_sm = tf.summary.image("plot", images, self.seq_len)

        cost_sm = tf.summary.scalar("cost", self.loss)
        cost_l2_sm = tf.summary.scalar("cost_l2", self.lossL2)
        self.merge_list = [image_sm,
                           cost_sm, cost_l2_sm]
        self.summarize = tf.summary.merge(self.merge_list)

        self.summarize_2 = tf.summary.merge([image_sm])
        self.increment_global_step_op = tf.assign(self.global_step, self.global_step + 1)

    def model(self):
        encoding = self.encoding(self.input_1)
        event_encoding = self.event_encoding(self.input_2)
        output = self.restructuring(encoding, event_encoding)
        return output

    def encoding(self, input):
        conv_lstm_1 = tf.contrib.rnn.ConvLSTMCell(conv_ndims=2,
                                                   input_shape=[self.size_x, self.size_y, 1],
                                                   output_channels=32,
                                                   kernel_shape=[5, 5],
                                                   use_bias=True)
        output_1, state_1 = tf.nn.dynamic_rnn(conv_lstm_1,
                                                   inputs=input,
                                                   dtype=tf.float32)

        conv_lstm_2 = tf.contrib.rnn.ConvLSTMCell(conv_ndims=2,
                                                  input_shape=[self.size_x, self.size_y, 32],
                                                  output_channels=32,
                                                  kernel_shape=[5, 5],
                                                  use_bias=True)
        empty_input = tf.zeros_like(input)
        output_2, state_2 = tf.nn.dynamic_rnn(conv_lstm_2,
                                                   empty_input,
                                                   initial_state=state_1,
                                                   dtype=tf.float32)
        output_2 = tf.reshape(output_2, [self.batch_size * self.seq_len, self.size_x, self.size_y, 32])
        return output_2

    def event_encoding(self, input):
        conv_lstm_fw = tf.contrib.rnn.ConvLSTMCell(conv_ndims=2,
                                                   input_shape=[self.size_x, self.size_y, 1],
                                                   output_channels=32,
                                                   kernel_shape=[5, 5],
                                                   use_bias=True)
        conv_lstm_bw = tf.contrib.rnn.ConvLSTMCell(conv_ndims=2,
                                                   input_shape=[self.size_x, self.size_y, 1],
                                                   output_channels=32,
                                                   kernel_shape=[5, 5],
                                                   use_bias=True)
        outputs, _ = tf.nn.bidirectional_dynamic_rnn(conv_lstm_fw, conv_lstm_bw,
                                                          inputs=input,
                                                          dtype=tf.float32)
        conv_lstm_2 = tf.contrib.rnn.ConvLSTMCell(conv_ndims=2,
                                                  input_shape=[self.size_x, self.size_y, 64],
                                                  output_channels=16,
                                                  kernel_shape=[5, 5],
                                                  use_bias=True)
        output_pre_softmax, _ = tf.nn.dynamic_rnn(conv_lstm_2,
                                                        inputs=tf.concat(outputs, -1),
                                                        dtype=tf.float32)

        maxpool_xy = tf.layers.max_pooling2d(output_pre_softmax, 8, 8, padding='valid')
        softmax_z = tf.nn.softmax(maxpool_xy, axis=-1)

        max_mask = tf.where(tf.equal(tf.reduce_max(softmax_z, axis=-1, keepdims=True), softmax_z),
                            tf.constant(1.0, shape=softmax_z.shape, dtype=tf.float32),
                            tf.constant(0.0, shape=softmax_z.shape, dtype=tf.float32))
        max_softmax_z = softmax_z * max_mask

        max_softmax_z = tf.reshape(max_softmax_z, [self.batch_size * self.seq_len, self.size_x/8, self.size_y/8, 16])

        shape = tf.convert_to_tensor([self.batch_size * self.seq_len, self.size_x, self.size_y, 32], dtype=tf.int32)
        filter = tf.ones([8, 8, 1, 1])
        upscaled_xy = tf.nn.conv2d_transpose(max_softmax_z, filter=filter,
                                         output_shape=shape,
                                         strides=[1, 8, 8, 1], padding='SAME')
        # upscaled_xy = tf.reshape(upscaled_xy, [self.batch_size, self.seq_len, self.size_x, self.size_y, 16])

        return upscaled_xy

    def restructuring(self, encoding, event_encoding):
        input = tf.concat((encoding, event_encoding), -1)
        conv1 = tf.layers.conv2d(input, filters=32, kernel_size=5, strides=(1, 1), padding='same', activation=tf.nn.sigmoid)
        conv2 = tf.layers.conv2d(conv1, filters=1, kernel_size=1, strides=(1, 1), padding='same', activation=tf.nn.sigmoid)
        conv2 = tf.reshape(conv2, [self.batch_size, self.seq_len, self.size_x, self.size_y, 1])
        return conv2

    def get_sample(self):
        # n = np.random.randint(0, self.training_data.shape[0] - 100 - self.seq_len, size=self.batch_size)
        n = np.random.randint(0, 20 - self.seq_len, size=self.batch_size)
        seq = np.transpose(np.array([self.training_data[n + i, :, :, :] for i in range(self.seq_len)])[:, :, :, :, :, 0], [1, 0, 2, 3, 4])
        labels = np.transpose(np.array([self.training_labels[n + i, :, :, :] for i in range(self.seq_len)]), [1, 0, 2, 3, 4])
        frames = np.transpose(np.array([self.training_frames[n + i, :, :, :] for i in range(self.seq_len)]), [1, 0, 2, 3, 4])
        # print seq.shape, labels.shape, frames.shape, 'seq, labels ============='
        return seq, labels, frames

    def run_train(self):
        with self.session.as_default(), self.session.graph.as_default():
            print 'started ---', self.epoch_size, 'epoch_size'
            self.gs = self.session.run(self.global_step)
            try:
                while self.gs <= 1000:
                    self.train_step()

                tf.logging.info("Reached global step {}. Stopping.".format(self.gs))
                self.saver.save(self.session, os.path.join(self.checkpoint_dir, 'my_model'), global_step=self.gs)
            except KeyboardInterrupt:
                print 'a du ----'
                self.saver.save(self.session, os.path.join(self.checkpoint_dir, 'my_model'), global_step=self.gs)
            return

    def train_step(self):
        start_time = time.time()
        # Get sample
        input, labels, frames = self.get_sample()

        feed_dict = {
            self.input: input,
            self.labels: labels,
            self.frames: frames,
        }
        loss, summary, _, self.gs = self.session.run([self.loss, self.summarize, self.train_op, self.global_step], feed_dict)
        duration = time.time() - start_time

        # emit summaries
        if self.gs % 10 == 9:
            print loss, duration, self.gs, 'loss, duration, gs'

        if self.gs % 10 == 9:
            print 'summary ---'
            self.summary_writer.add_summary(summary, self.gs)

        if self.gs % 500 == 100 and self.gs > 100:
            print("Saving model checkpoint: {}".format(str(self.gs)))
            self.saver.save(self.session, os.path.join(self.checkpoint_dir, 'my_model'), global_step=self.gs)

    def get_inference_frames(self, x=0):
        more_frames = True
        x_batch = np.arange(x, np.minimum(x + self.batch_size, self.training_data.shape[0] - self.seq_len))
        seq = np.transpose(np.array([self.training_data[x_batch + i, :, :, :] for i in range(self.seq_len)])[:, :, :, :, :, 0], [1, 0, 2, 3, 4])
        labels = np.transpose(np.array([self.training_labels[x_batch + i, :, :, :] for i in range(self.seq_len)]), [1, 0, 2, 3, 4])
        frames = np.transpose(np.array([self.training_frames[x_batch + i, :, :, :] for i in range(self.seq_len)]), [1, 0, 2, 3, 4])
        shape = frames.shape
        zeros_size = 0
        if shape[0] < self.batch_size:
            seq = np.concatenate((seq, np.zeros([self.batch_size - seq.shape[0]] + list(seq.shape[1:]))), 0)
            frames = np.concatenate((frames, np.zeros([self.batch_size - frames.shape[0]] + list(frames.shape[1:]))), 0)
            labels = np.concatenate((labels, np.zeros([self.batch_size - labels.shape[0]] + list(labels.shape[1:]))), 0)
            more_frames = False
            zeros_size = self.batch_size - shape[0]
        return seq, labels, frames, x + self.batch_size, more_frames, zeros_size

    def get_centers(self, map):
        map = binary_erosion(map, structure=np.ones((1,3,3))).astype(map.dtype)
        map = filters.gaussian_filter(map, sigma=1)
        map[map > 0] = 1.0
        labels, number = label(map)
        map_points = np.zeros_like(map)
        for n in range(number):
            n += 1
            m = deepcopy(labels)
            m[m != n] = 0
            c = center_of_mass(m)
            c = np.round(c).astype(np.int32)
            # print c, 'c'
            map_points[c[0], c[1], c[2]] = n + 1
        return map_points

    def inference(self, inference_size=100):
        more_frames = True
        x = 0
        big_output = np.zeros([inference_size, self.size_x, self.size_y])

        self.training_data = self.training_data[-inference_size:, :, :, :]
        self.training_frames = self.training_frames[-inference_size:, :, :, :]
        self.training_labels = self.training_labels[-inference_size:, :, :, :]

        all_labels = self.ground_truth_matrix(compress_jump=4, z_tolerance=3, xy_tolerance=20)
        big_labels = all_labels[-inference_size:, :, :]
        big_labels[:self.z_margin, :, :] = 0
        big_labels[-self.z_margin:, :, :] = 0

        zeros_size = 0
        while more_frames:
            print x, 'x'
            seq, labels, frames, x, more_frames, zeros_size = self.get_inference_frames(x)
            # x -= self.z_margin * 2

            start_time = time.time()
            feed_dict = {
                self.input: seq,
                self.labels: labels,
                self.frames: frames,
            }

            output, summary, _, self.gs = self.session.run([self.output, self.summarize_2, self.increment_global_step_op, self.global_step], feed_dict)
            duration = time.time() - start_time
            print duration, 'sec', self.gs
            self.summary_writer.add_summary(summary, self.gs)

            output = output[:, self.z_margin, :, :, 0]
            big_output[x - self.batch_size + self.z_margin:x + self.z_margin, :, :] = output
            if zeros_size > 0:
                big_output[-zeros_size:, :, :] = 0

        big_output[big_output > 0.5] = 1
        big_output[big_output <= 0.5] = 0
        center_pred = self.get_centers(big_output)
        no_detected = len(np.unique(center_pred)) - 1
        print np.unique(center_pred), 'center pred'

        # center_label = self.get_centers(big_labels)
        no_mitosis = 240  # len(np.unique(center_label)) - 1
        # print np.unique(center_label), 'center label'

        correct_pred = center_pred * big_labels
        TP = float(len(np.unique(correct_pred)) - 1)
        FP = float(no_detected - TP)
        FN = float(no_mitosis - TP)

        print TP, FP, FN, 'TP, FP, FN'
        precision = (TP / (TP + FP)) * 100
        recall = (TP / (TP + FN)) * 100
        print precision, recall, 'precision, recall'
        F1_score = 2 * (precision * recall / (precision + recall))
        print precision, recall, F1_score, 'precision, recall, F1_score'

    def read_ground_truth(self, file_path):
        with open(file_path) as f:
            lines = f.readlines()
        # output = np.zeros((len(lines), 3))
        output = None
        skip = 0  # 111
        for i in range(len(lines)):
            content = lines[i].split(' ')
            content = [np.round(float(x)) for x in content]
            if int(float(content[0])) >= skip:
                # print content[0]
                if output is None:
                    a = np.array(content[:3], dtype='float32')
                    a[0] = a[0] - skip
                    output = a
                else:
                    a = np.array(content[:3], dtype='float32')
                    a[0] = a[0] - skip
                    output = np.vstack((output, a))
        return output

    def ground_truth_matrix(self, compress_jump=4, z_tolerance=1, xy_tolerance=10):
        gt_file = '/media/newhd/Ha/data/BAEC/F0001/BAEC_seq1_mitosis.txt'
        groundtruth = self.read_ground_truth(gt_file)
        shape = self.training_frames.shape
        max_z = 210; max_x = 1392; max_y = 1040
        matrix = np.zeros([max_z, max_y, max_x])
        for i in range(groundtruth.shape[0]):
            z = int(groundtruth[i, 0] - 1)
            x = int(groundtruth[i, 1])
            y = int(groundtruth[i, 2])
            z0 = z - z_tolerance if z - z_tolerance > 0 else 0
            z1 = z + z_tolerance + 1 if z + z_tolerance + 1 < max_z else max_z
            x0 = x - xy_tolerance if x - xy_tolerance >= 0 else 0
            x1 = x + xy_tolerance + 1 if x + xy_tolerance + 1 <= max_x else max_x
            y0 = y - xy_tolerance if y - xy_tolerance >= 0 else 0
            y1 = y + xy_tolerance + 1 if y + xy_tolerance + 1 <= max_y else max_y
            z0 = int(z0); z1 = int(z1); y0 = int(y0); y1 = int(y1); x0 = int(x0); x1 = int(x1)
            matrix[z0:z1, y0:y1, x0:x1] = 1.0
            # matrix[z, y:y+compress_jump, x:x+compress_jump] = 1.0
        if True:
            matrix = matrix[:, ::compress_jump, ::compress_jump]
        return matrix

