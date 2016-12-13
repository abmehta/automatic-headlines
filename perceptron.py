import read_rcv1 as read
import math
import operator
import random
import itertools
from collections import defaultdict

weights = defaultdict(float)
weight_feats = defaultdict(float)


def main(path,mode):
	
	docs = read.get_split_data(path)
	prob = {}
	topn = {}
	count = 0
	for doc in docs:
		count += 1			#temporary to keep a count of the documents processed
		for index in range(len(doc.text_pos)):
			feature_vec, outcome = doc.get_local_feature(index)
			if mode == 'train':	
				train(feature_vec,outcome)
			elif mode == 'test':
				prob[doc.text_pos[index][0]] = test(feature_vec)
		if mode == 'test':	
			topn[doc.doc_id] = sorted(prob.items(), key=operator.itemgetter(0), reverse=True)[:20]	#randomly put 20 here
		print "count", count		#temporary
	return topn

def train(feature_vec,outcome,learning_rate=0.001):			#have ignored validation set and set learning_rate arbitrarily

	output = predict(feature_vec)
	error = outcome - output
	for k in feature_vec.keys():
		weights[k] += learning_rate*error

def predict(feature_vec):
    
	expo_sum = 0
	for key,value in feature_vec.iteritems():
		if weights[key] == 0.0:
			weights[key] = random.uniform(0, 1)
		weight_feats[key] = weights[key]*value
	normalize = max(weight_feats.values())
	for keys,value in weight_feats.iteritems():
		weight_feats[key] = value-normalize
	for key in feature_vec.keys():
		expo_sum += math.exp(weight_feats[key])
	output = math.log(expo_sum)/expo_sum
	return output
	
def test(feature_vec):
    	output = predict(feature_vec)
	return output

if __name__ == "__main__":
	
	main('data/train_sample365.split','train')		#small files that I made created with 1 file a day instead of 100
	topn = main('data/test_sample365.split','test')
	print "topn",topn
	



