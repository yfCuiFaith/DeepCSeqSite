import numpy as np
import math
import os
import termios
import time
import sys

def CurrentTime():
	return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))

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

def ReadDataSet(data_dir, feature_file, label_file, dataset_type, dataset_name = None):
	if(dataset_name != None):
		print('\nReading %s dataset...' % (dataset_name))

	dataset = dataset_type(data_dir + feature_file, \
								data_dir + label_file)

	print('%s size = %d' % (dataset_name, dataset.number))

	if(dataset_name != None):
		print('Reading %s dataset complete.\n' % (dataset_name))

	return dataset

# pred: prediction, 0 or 1
# rw: right or wrong, bool
def Statistic(pred, rw):
	if(len(pred) != len(rw)):
		return None
	
	tp = 0.
	tn = 0.
	fp = 0.
	fn = 0.
	
	for i in range(0, len(rw)):
		if(rw[i]):
			if(pred[i] == 1):
				tp += 1
			else:
				tn += 1
		else:
			if(pred[i] == 1):
				fp += 1
			else:
				fn += 1
	return tp, tn, fp, fn

def GetMCC(tp, tn, fp, fn):
	deno = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
	if(deno):
		return (tp * tn - fp * fn) / deno
	else:
		return 0.0

class HistoryPool:
	def __init__(self, init_lengths):
		self.index = 0
		self.pool = []
		self.lengths = init_lengths
		self.size = len(init_lengths)

		for i in range(len(init_lengths)):
			self.pool.append([[[0.0, 0.0]] for j in range(init_lengths[i])])
	
	def Forget(self):
		for i in range(len(self.pool)):
			self.pool[i] = [[[0.0, 0.0]] for j in range(len(self.pool[i]))]

	def Update(self, update_tensor):
		batch_size = len(update_tensor)
		for i in range(0, batch_size):
			self.pool[self.index] = update_tensor[i][:self.lengths[self.index]]
			self.index = (self.index + 1) % self.size

	def Get(self, batch_size, filler):
		lengths = [len(self.pool[i % self.size]) \
			for i in range(self.index, self.index + batch_size)]
		max_len = max(lengths)

		res = []
		for i in range(batch_size):
			pad_size = max_len - lengths[i]

			prim_tensor = self.pool[(self.index + i) % self.size]
			padding = [filler for j in range(pad_size)]

			if(isinstance(prim_tensor, np.ndarray)):
				prim_tensor = prim_tensor.tolist()

			res.append(prim_tensor + padding)

		return res

	def GoAhead(self, batch_size):
		self.index = (self.index + batch_size) % self.size
