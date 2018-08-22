import tensorflow as tf
import os
import sys
import termios
from model import Model

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_boolean('interact', True, \
	'If True, the program will be executed interactively')
flags.DEFINE_string('testset_dir', '', 'The dir of test set')

model_dir = 'width_k5/'

model_name = 'model.ckpt'

SITA_dir = 'DataSet/SITA/'
SITA_EX1_dir = 'DataSet/SITA_EX1/'
SITA_EX2_dir = 'DataSet/SITA_EX2/'
SITA_EX3_dir = 'DataSet/SITA_EX3/'

batch_size = 1

softmax_thr = 0.2

def Session():
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	sess = tf.Session(config = config)
	return sess 

def getche():
	fd = sys.stdin.fileno()

	old_ttyinfo = termios.tcgetattr(fd)

	new_ttyinfo = old_ttyinfo[:]

	new_ttyinfo[3] &= ~termios.ICANON
	new_ttyinfo[3] &= ~termios.ECHO

	termios.tcsetattr(fd, termios.TCSANOW, new_ttyinfo)

	buf = os.read(fd, 1)

	termios.tcsetattr(fd, termios.TCSANOW, old_ttyinfo)

	return buf

def main(_):
	dcs_si = Model(model_dir, model_name, batch_size, softmax_thr)

	sess = Session()

	dcs_si.restore_parameters(sess)

	while(True):
		print('Please input one of the following section id:')
		print('1. Test SITA')
		print('2. Test SITA-EX1')
		print('3. Test SITA-EX2')
		print('4. Test SITA-EX3')
		print('5. Exit')

		buf = getche().strip()

		if(buf == '1'):
			dcs_si.benchmark(sess, SITA_dir, 'SITA')
			continue

		if(buf == '2'):
			dcs_si.benchmark(sess, SITA_EX1_dir, 'SITA_EX1')
			continue

		if(buf == '3'):
			dcs_si.benchmark(sess, SITA_EX2_dir, 'SITA_EX2')
			continue

		if(buf == '4'):
			dcs_si.benchmark(sess, SITA_EX3_dir, 'SITA_EX3')
			continue

		if(buf == '5'):
			print('Exit')
			break

		print('\n')

if __name__ == '__main__':
	tf.app.run()
