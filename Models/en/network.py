import tensorflow as tf
from tf_lib import WeightVariable
from tf_lib import BiasVariable
from tf_lib import Conv
from tf_lib import BatchNorm
from tf_lib import LayerNorm
from tf_lib import GLU
from tf_lib import ResidualBlock
from tf_lib import PlainBlock
from tf_lib import NormBlock
from tf_lib import SeqLeftShift
from tf_lib import SeqRightShift
from tf_lib import DevConv
from tf_lib import DevResidualBlock
from tf_lib import DevPlainBlock

kernel_width = 5 # Height
amino_dim = 30
std_in_channel = 256
std_out_channel = std_in_channel * 2

std_filter_shape = [kernel_width, 1, std_in_channel, std_out_channel]
hist_filter_shape = [3, 1, 2, std_out_channel]

#padding = kernel_width / 2

def Inference(x, y, batch_size, keep_prob):

	with tf.name_scope('norm'):
		x = tf.reshape(x, [batch_size, -1, amino_dim, 1])

	with tf.name_scope('encoder'):
		with tf.name_scope('trans'):
			conv0_filter_shape = [3, amino_dim, 1, std_out_channel]

			conv0_input = tf.pad(x, [[0, 0], [1, 1], [0, 0], [0, 0]])
			conv0 = Conv(conv0_input, conv0_filter_shape, name = 'trans_conv')

		buffer_tensor = conv0

		with tf.name_scope('stage1'):
			for i in range(0, 10):
				block_scope = str.format('stage1_block%d' % (i))

				buffer_tensor = \
					ResidualBlock(buffer_tensor, std_filter_shape, block_scope)
				#print buffer_tensor
			buffer_tensor = \
				PlainBlock(buffer_tensor, std_filter_shape, 'stage1_top')
			#print buffer_tensor

		with tf.name_scope('stage2'):
			for i in range(0, 10):
				block_scope = str.format('stage2_block%d' % (i))

				buffer_tensor = \
					ResidualBlock(buffer_tensor, std_filter_shape, block_scope)
				#print buffer_tensor
				
			buffer_tensor = \
				NormBlock(buffer_tensor, 'stage2_top')
			#print buffer_tensor
		
		encoder_output = buffer_tensor

	with tf.name_scope('summary'):
		info_l = SeqRightShift(y)
		info_l = DevConv(info_l, hist_filter_shape, 'L', 'L_conv0')

		for i in range(0, 2):
			block_scope = str.format('L_block%d' % (i))

			info_l = DevResidualBlock(info_l, std_filter_shape, 'L', block_scope)

		info_l = NormBlock(info_l, 'L_top')
		
		info_r = SeqLeftShift(y)
		info_r = DevConv(info_r, hist_filter_shape, 'R', 'R_conv0')

		for i in range(0, 2):
			block_scope = str.format('R_block%d' % (i))

			info_r = DevResidualBlock(info_r, std_filter_shape, 'R', block_scope)

		info_r = NormBlock(info_r, 'R_top')

		decoder_output = encoder_output + info_l + info_r

	with tf.name_scope('proj'):
		fc0_filter_shape = [1, 1, std_in_channel, std_in_channel]
		fc0 = Conv(decoder_output, fc0_filter_shape, bias = True, name = 'fc0_conv')
		fc0_relu = tf.nn.relu(fc0)
		fc0_drop = tf.nn.dropout(fc0_relu, keep_prob)
		#print fc0_drop

		fc1_filter_shape = [1, 1, std_in_channel, 2]
		fc1 = Conv(fc0_drop, fc1_filter_shape, bias = True, name = 'fc1_conv')
		fc1_relu = tf.nn.relu(fc1)
		#print fc1_relu

	return fc1_relu, info_l, info_r

