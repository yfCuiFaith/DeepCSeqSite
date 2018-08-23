import tensorflow as tf
import sys
import importlib
from tf_lib import PredictionResult
from seq_dataset import SeqDataSet
from toolkit import ReadDataSet
from toolkit import Statistic
from toolkit import GetMCC

class Model:
	def mask_padding(self, src, lens, batch_size):
		res = src[0][:lens[0]]

		for i in range(1, batch_size):
			res = tf.concat([res, src[i][:lens[i]]], 0)
		return res

	def define_model(self):
		with tf.variable_scope('param'):
			self.keep_prob = tf.placeholder(tf.float32, name = 'keep_prob')

		with tf.name_scope('input'):
			input_x = tf.placeholder(tf.float32, [self.batch_size, None], name = 'fature_map')
			input_y = tf.placeholder(tf.float32, [self.batch_size, None, 2], name = 'label')
			ly = tf.placeholder(tf.int32, [self.batch_size], name = 'ly')

		with tf.name_scope('forward'):
			x_ = self.network.Inference(input_x, self.batch_size, self.keep_prob)
			x_ = tf.reshape(x_, [self.batch_size, -1, 2])
			re_x_ = self.mask_padding(x_, ly, self.batch_size)

		with tf.name_scope('softmax'):
			y_ = tf.nn.softmax(re_x_)
		
		return input_x, input_y, ly, y_

	def assess(self, pred_y, input_y, lens_y):
		grd_y = self.mask_padding(input_y, lens_y, self.batch_size)
		pred_analy = PredictionResult(pred_y, grd_y, self.softmax_thr)

		return pred_analy

	def __init__(self, model_dir, batch_size, softmax_thr):
		self.model_dir = model_dir
		self.model_name = tf.train.latest_checkpoint(model_dir)
		self.batch_size = batch_size
		self.softmax_thr = softmax_thr

		sys.path.append(model_dir)
		self.network = importlib.import_module('network')

		self.input_x, self.input_y, self.len_y, self.pred_y \
			= self.define_model()

		self.pred_analy = self.assess(self.pred_y, self.input_y, self.len_y)

	def restore_parameters(self, sess):
		saver = tf.train.Saver()
		
		saver.restore(sess, self.model_name)
		print('\nModel has been loaded.\n')

	def test(self, sess, dataset):
		total_prec = 0.0
		total_reca = 0.0
		total_mcc = 0.0
		total_iter = 0

		dataset.SetSignal()

		while(True):
			batch_x, batch_y, len_x, len_y = dataset.NextRestrictedPaddingBatch(self.batch_size)

			if(batch_y is None):
				break;

			input_dict = {}

			input_dict[self.input_x] = batch_x
			input_dict[self.input_y] = batch_y
			input_dict[self.len_y] = len_y
			input_dict[self.keep_prob] = 1.0

			pred_rw_acc = sess.run(self.pred_analy, feed_dict = input_dict)

			pred = pred_rw_acc[0]
			rw = pred_rw_acc[1]
			acc = pred_rw_acc[2]

			tp, tn, fp, fn = Statistic(pred, rw)
			precision = tp / (tp + fp + 0.001)
			recall = tp / (tp + fn + 0.001)
			mcc = GetMCC(tp, tn, fp, fn)

			# protein level estimation
			total_prec += precision
			total_reca += recall
			total_mcc += mcc
			total_iter += 1

		print('Precision = %.2f%%' % (total_prec * 100 / total_iter))
		print('Recall = %.2f%%' % (total_reca * 100 / total_iter))
		print('MCC = %.3f' % (total_mcc / total_iter))
		print('\n')

	def benchmark(self, sess, dataset_dir, dataset_name):
		dataset = ReadDataSet(dataset_dir, 'data.feat', 'data.lab', SeqDataSet, dataset_name)

		self.test(sess, dataset)

		del dataset
