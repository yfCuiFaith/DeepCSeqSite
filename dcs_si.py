import tensorflow as tf
import sys
from model import Model
from toolkit import getche

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_boolean('interact', True, \
	'If True, the program will be executed interactively')
flags.DEFINE_string('model', 'DCS-SI-std', \
	'Options: DCS-SI-std, DCS-SI-k9, DCS-SI-k9a')
flags.DEFINE_string('testset_dir', '', 'The dir of test set')

model_root_dir = 'Models'

model_dict = {'DCS-SI-std': 'std/', \
			'DCS-SI-k9': 'k9/', \
			'DCS-SI-k9a': 'k9a/'}

dataset_root_dir = 'DataSet'

dataset_list = ['SITA', 'SITA_EX1', 'SITA_EX2', 'SITA_EX3']


batch_size = 1

softmax_thr = 0.2

def Session():
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	sess = tf.Session(config = config)
	return sess 

def main(_):
	if(model_dict.has_key(FLAGS.model) != True):
		print('Model %s is not existed.' % FLAGS.model)
		print('Options: DCS-SI-std, DCS-SI-k9, DCS-SI-k9a\n')
		sys.exit()

	model_dir = 'Models/%s' % model_dict[FLAGS.model]

	dcs_si = Model(model_dir, batch_size, softmax_thr)

	sess = Session()

	dcs_si.restore_parameters(sess)

	while(True):
		print('Please input one of the following option id:')
		print('1. Test SITA')
		print('2. Test SITA-EX1')
		print('3. Test SITA-EX2')
		print('4. Test SITA-EX3')
		print('5. Exit')

		buf = getche().strip()

		if(buf == '1'):
			dataset_dir = '%s/SITA/' % dataset_root_dir
			dcs_si.benchmark(sess, dataset_dir, 'SITA')
			continue

		if(buf == '2'):
			dataset_dir = '%s/SITA_EX1/' % dataset_root_dir
			dcs_si.benchmark(sess, dataset_dir, 'SITA_EX1')
			continue

		if(buf == '3'):
			dataset_dir = '%s/SITA_EX2/' % dataset_root_dir
			dcs_si.benchmark(sess, dataset_dir, 'SITA_EX2')
			continue

		if(buf == '4'):
			dataset_dir = '%s/SITA_EX3/' % dataset_root_dir
			dcs_si.benchmark(sess, dataset_dir, 'SITA_EX3')
			continue

		if(buf == '5'):
			print('Exit')
			break

		print('\n')

if __name__ == '__main__':
	tf.app.run()
