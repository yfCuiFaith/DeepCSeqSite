# list only
class SeqDataSet:
	def ReadFeatureSource(self, src):
		if(isinstance(src, list)):
			return src

		with open(src) as srcFile:
			buf = srcFile.readlines()

		for i in range(len(buf)):
			buf[i] = buf[i].strip().split(' ')
			for j in range(len(buf[i])):
				buf[i][j] = float(buf[i][j])

		return buf

	def ReadLabelSource(self, src):
		if(isinstance(src, list)):
			return src

		with open(src) as srcFile:
			buf = srcFile.readlines()

		for i in range(len(buf)):
			buf[i] = list(buf[i].strip())
			for j in range(len(buf[i])):
				if(buf[i][j] == '0'):
					buf[i][j] = [1.0, 0.0]
				else:
					buf[i][j] = [0.0, 1.0] 
		
		return buf

	def __init__(self, featureSource, labelSource):
		self.feature = self.ReadFeatureSource(featureSource)
		self.label = self.ReadLabelSource(labelSource)
		self.index = 0

		if(len(self.feature) != len(self.label)):
			self.number = -1
			print('###DataError!###')
		else:
			self.number = len(self.label)

	def Fill(self, src_seqs, filler):
		seqs_lens = [len(seq) for seq in src_seqs]

		batch_size = len(src_seqs)

		max_len = max(seqs_lens)

		dest_seqs = []
		for i in range(0, batch_size):
			pad_size = max_len - seqs_lens[i]
			dest_seqs.append(src_seqs[i] + \
				[filler for j in range(pad_size)])
		
		return dest_seqs, seqs_lens

	def NextBatch(self, batch_size = 1):
		feature_batch = None
		label_batch = None
		
		if(self.index + batch_size <= self.number):
			feature_batch = \
				self.feature[self.index: self.index + batch_size]
			label_batch = \
				self.label[self.index: self.index + batch_size]
			self.index = self.index + batch_size
			
			if(self.index == self.number):
				self.index = 0
			
			return feature_batch, label_batch
		
		next_end = (self.index + batch_size) % self.number
		
		feature_batch = \
			self.feature[self.index:] + self.feature[:next_end]
		label_batch = \
			self.label[self.index:] + self.label[:next_end]
		self.index = next_end
		
		return feature_batch, label_batch

	def NextPaddingBatch(self, batch_size = 1):
		batch_x, batch_y = self.NextBatch(batch_size)

		batch_x, len_x = self.Fill(batch_x, 0.0)
		batch_y, len_y = self.Fill(batch_y, [0.0, 0.0])

		return batch_x, batch_y, len_x, len_y

	def SetSignal(self):
		self.avail_num = self.number

	# |dataset| % batch_size must be 0
	def NextRestrictedBatch(self, batch_size = 1):
		if(self.avail_num > 0):
			self.avail_num -= batch_size
			return self.NextBatch(batch_size)
		return [None, None]

	def NextRestrictedPaddingBatch(self, batch_size = 1):
		if(self.avail_num > 0):
			batch_size = min(self.avail_num, batch_size)
			self.avail_num -= batch_size
			return self.NextPaddingBatch(batch_size)
		return [None, None, None, None]

	def EntireBatch(self):
		return self.feature, self.label
