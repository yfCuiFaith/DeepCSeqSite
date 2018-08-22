import numpy as np
import math
import time
import gc

def CurrentTime():
	return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))

def ReadDataSet(data_dir, feature_file, label_file, dataset_type, dataset_name = None):
	if(dataset_name != None):
		print('Reading %s dataset...' % (dataset_name))

	dataset = dataset_type(data_dir + feature_file, \
								data_dir + label_file)

	print('%s size = %d' % (dataset_name, dataset.number))

	if(dataset_name != None):
		print('Reading %s dataset complete.' % (dataset_name))

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
