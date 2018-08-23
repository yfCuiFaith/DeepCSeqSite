import tensorflow as tf
import sys
sys.path.append('..')

from tf_lib import WeightVariable
from tf_lib import BiasVariable
from tf_lib import Conv
from tf_lib import BatchNorm
from tf_lib import LayerNorm
from tf_lib import GLU
from tf_lib import ResidualBlock
from tf_lib import PlainBlock
from tf_lib import NormBlock

kernel_width = 9 # Height
amino_dim = 30
std_in_channel = 256
std_out_channel = std_in_channel * 2

std_filter_shape = [kernel_width, 1, std_in_channel, std_out_channel]
attn_filter_shape = [kernel_width, 1, std_in_channel, std_in_channel]

#padding = kernel_width / 2

def Inference(x, batch_size, keep_prob):

	with tf.name_scope('norm'):
		x = tf.reshape(x, [batch_size, -1, amino_dim, 1])

	with tf.name_scope('trans'):
		conv0_filter_shape = [3, amino_dim, 1, std_out_channel]

		conv0_input = tf.pad(x, [[0, 0], [1, 1], [0, 0], [0, 0]])
		conv0 = Conv(conv0_input, conv0_filter_shape, name = 'trans_conv')

		#print conv0
	
	with tf.name_scope('stage1'):
		buffer_tensor = conv0
		for i in range(0, 10):
			block_scope = str.format('stage1_block%d' % (i))

			buffer_tensor = \
				ResidualBlock(buffer_tensor, std_filter_shape, block_scope)
			#print buffer_tensor
		buffer_tensor = \
			PlainBlock(buffer_tensor, std_filter_shape, 'stage1_top')
		#print buffer_tensor

	encoder_output = buffer_tensor

	with tf.name_scope('stage2'):
		for i in range(0, 10):
			block_scope = str.format('stage2_block%d' % (i))

			buffer_tensor = \
				ResidualBlock(buffer_tensor, std_filter_shape, block_scope)
			#print buffer_tensor

		buffer_tensor = \
			NormBlock(buffer_tensor, 'stage2_top')
		#print buffer_tensor
	
	with tf.name_scope('proj'):
		fc0_filter_shape = [1, 1, std_in_channel, std_in_channel]
		fc0 = Conv(buffer_tensor, fc0_filter_shape, bias = True, name = 'fc0_conv')
		fc0_relu = tf.nn.relu(fc0)
		fc0_drop = tf.nn.dropout(fc0_relu, keep_prob)
		#print fc0_drop
    
		fc1_filter_shape = [1, 1, std_in_channel, 2]
		fc1 = Conv(fc0_drop, fc1_filter_shape, bias = True, name = 'fc1_conv')
		fc1_relu = tf.nn.relu(fc1)
		#print fc1_relu

	return fc1_relu

