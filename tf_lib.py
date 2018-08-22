import tensorflow as tf

def WeightVariable(shape, scope = None, var_name = None, regularizer = 'l2_weights'):
	weight = tf.Variable(tf.truncated_normal(shape, stddev = 0.1), \
			name = var_name)

	if(regularizer is not None):
		tf.add_to_collection(regularizer, weight)

	return weight

def BiasVariable(value, shape, scope = None, var_name = None):
	return tf.Variable(tf.constant(value, shape = shape), name = 'bias')

def NormVariable(shape, scope = None, ofs_name = None, sc_name = None):
	offset = tf.Variable(tf.zeros(shape), name = ofs_name)
	scale = tf.Variable(tf.ones(shape), name = sc_name)
	return offset, scale

def Conv(input_tensor, filter_shape, strides = [1, 1, 1, 1], padding = 'VALID', \
		bias = False, data_format = 'NHWC', name = None):
	# name
	scope = name + '_w'
	filter_name = 'filter'
	bias_name = 'bias'

	# conv
	conv_filter = WeightVariable(filter_shape, scope, filter_name)

	conv_res = tf.nn.conv2d(input_tensor, conv_filter, strides = strides, \
				padding = padding, data_format = data_format, name = name)

	# bias
	if(bias is False):
		return conv_res
	else:
		if(data_format == 'NHWC'):
			channel_num = filter_shape[3]
		else: #'NCHW'
			channel_num = filter_shape[1]

		conv_bias = BiasVariable(0.1, [channel_num], scope, bias_name)

		return tf.nn.bias_add(conv_res, conv_bias)

def FC(input_tensor, weights, biases):
	return tf.matmul(input_tensor, weights) + biases

def GLU(input_tensor):
	tensor_a, tensor_b = tf.split(input_tensor, 2, 3)
	return tensor_a * tf.nn.sigmoid(tensor_b)

def BatchNorm(input_tensor, shape, is_test, iteration, name = None):
	# name
	scope = name + '_w'
	offset_name = 'offset'
	scale_name = 'scale'

	offset, scale = NormVariable(shape, scope, offset_name, scale_name)
	ema = tf.train.ExponentialMovingAverage(0.998, iteration)

	mean, variance = tf.nn.moments(input_tensor, [0])
	update_moving_average = ema.apply([mean, variance])

	m = tf.cond(is_test, lambda: ema.average(mean), lambda: mean)
	v = tf.cond(is_test, lambda: ema.average(variance), lambda: variance)

	output_tensor = \
		tf.nn.batch_normalization(input_tensor, m, v, offset, scale, 0.001)

	return output_tensor, update_moving_average

def LayerNormMoments(x, axes = 1, scope = None, epsilon = 0.001):
	with tf.name_scope(scope):
		mean = tf.reduce_mean(x, axes, keepdims = True)

		variance = tf.sqrt(tf.reduce_mean(\
			tf.square(x - mean), axes, keepdims = True) + epsilon)

	return mean, variance

# input_tensor: [m, d]
# offset: [d]
# scale: [d]
# output_tensor: [m, d]
def LayerNorm(input_tensor, scope):
	with tf.name_scope(scope):
		input_tensor_shape = input_tensor.get_shape().as_list()
		ln_shape = [input_tensor_shape[1]]

		offset, scale = NormVariable(ln_shape, scope, 'offset', 'scale')
		mean, variance = LayerNormMoments(input_tensor)

		output_tensor = (scale * (input_tensor - mean)) / variance + offset

	return output_tensor

# pred: tensor of prediction
# grd: tensor of ground truth
def PredictionResult(pred, grd, threshold = None):
	if(threshold):
		ofs_value = 0.5 - threshold
		ofs = tf.constant([-ofs_value, ofs_value], dtype = tf.float32)
		pred = pred + ofs
	
	prediction = tf.argmax(pred, 1)
	ground_truth = tf.argmax(grd, 1)
	correctness = tf.equal(prediction, ground_truth)
	accuracy = tf.reduce_mean(tf.cast(correctness, tf.float32))
	return prediction, correctness, accuracy

def PlainBlock(input_tensor, filter_shape, block_name):
	with tf.name_scope(block_name):
		input_tensor_shape = input_tensor.get_shape().as_list()
		batch_size = input_tensor_shape[0]

		padding = filter_shape[0] / 2

		conv_name = '%s_conv' % block_name
		ln_name = '%s_ln' % block_name

		block_input = tf.reshape(input_tensor, [-1, input_tensor_shape[3]])
		block_ln = LayerNorm(block_input, scope = ln_name)
		block_ln = tf.reshape(block_ln, [batch_size, -1, 1, input_tensor_shape[3]])
		block_glu = GLU(block_ln)

		block_glu = tf.pad(block_glu, [[0, 0], [padding, padding], [0, 0], [0, 0]])

		block_conv = Conv(block_glu, filter_shape, name = conv_name)

	return block_conv

def ResidualBlock(input_tensor, filter_shape, block_name):
	with tf.name_scope(block_name):
		input_tensor_shape = input_tensor.get_shape().as_list()
		batch_size = input_tensor_shape[0]

		padding = filter_shape[0] / 2

		conv_name = '%s_conv' % block_name
		ln_name = '%s_ln' % block_name

		block_input = tf.reshape(input_tensor, [-1, input_tensor_shape[3]])
		block_ln = LayerNorm(block_input, scope = ln_name)
		block_ln = tf.reshape(block_ln, [batch_size, -1, 1, input_tensor_shape[3]])
		block_glu = GLU(block_ln)
		
		block_glu = tf.pad(block_glu, [[0, 0], [padding, padding], [0, 0], [0, 0]])

		block_conv = Conv(block_glu, filter_shape, name = conv_name)

		block_output = input_tensor + block_conv
	return block_output

def NormBlock(input_tensor, block_name):
	with tf.name_scope(block_name):
		input_tensor_shape = input_tensor.get_shape().as_list()
		batch_size = input_tensor_shape[0]

		conv_name = '%s_conv' % block_name
		ln_name = '%s_ln' % block_name

		block_input = tf.reshape(input_tensor, [-1, input_tensor_shape[3]])
		block_ln = LayerNorm(block_input, scope = ln_name)
		block_ln = tf.reshape(block_ln, [batch_size, -1, 1, input_tensor_shape[3]])
		block_glu = GLU(block_ln)
	return block_glu

