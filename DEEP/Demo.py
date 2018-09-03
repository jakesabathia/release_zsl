import tensorflow as tf
import numpy as np
import scipy.io as sio
import sys
import math
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import matplotlib

class model():
	def __init__(self,n_feature,n_semantic,n_latent):
		self.lr = 0.01
		#input
		self.image_input = tf.placeholder(tf.float32,shape=(None,n_feature))
		self.semantic_input = tf.placeholder(tf.float32,shape=(None,n_semantic))
		self.label_input = tf.placeholder(tf.int32,shape=(None,))
		
		#image
		self.latent_image = tf.layers.batch_normalization(self.image_input)
		self.latent_image = tf.layers.dense(self.latent_image,400,use_bias=False)
		self.latent_image = tf.nn.sigmoid(self.latent_image)
		self.latent_image = tf.layers.batch_normalization(self.image_input)
		self.latent_image = tf.layers.dense(self.latent_image,n_latent)
		self.latent_image = tf.nn.sigmoid(self.latent_image)
		
		
		self.latent_semantic = tf.layers.dense(self.semantic_input,n_latent,use_bias=False)
		self.logits = tf.matmul(self.latent_image,self.latent_semantic,transpose_b=True)
		self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.label_input,logits=self.logits)
		self.final_loss = tf.reduce_mean(self.loss)
		self.learner = tf.train.AdagradOptimizer(learning_rate=self.lr).minimize(self.final_loss)
		self.predict_label = tf.argmax(self.logits,axis=1)
		print('build model')
		
if __name__ == '__main__':
	
	#Load Image and Semantic
	Data_path = './Data/'
	dataset = 'AWA' #AWA, CUB, DOG
	semantic = 'cont' # cont, word2vec, glove, wordnet
	Data_name = 'data_'+dataset
	s_Data_name = dataset+'_s'
	raw_data = sio.loadmat(Data_path+Data_name)
	raw_s_data = sio.loadmat(Data_path+s_Data_name)
	coco = raw_data['train_X']
	train_Y = raw_s_data['train_'+semantic]
	train_X = raw_data['train_X']
	train_labels = raw_data['train_labels']
	test_X = raw_data['test_X']
	test_Y = raw_s_data['test_'+semantic]
	test_labels = raw_data['test_labels']
	train_labels = train_labels-1
	test_labels = test_labels-1
	train_labels = np.squeeze(train_labels)
	test_labels = np.squeeze(test_labels)
	train_Y = train_Y.T
	test_Y = test_Y.T
	
	#build ZSL model
	n_feature = train_X.shape[1]
	n_semantic = train_Y.shape[1]
	n_latent = int(n_semantic)
	n_data = train_X.shape[0]
	model = model(n_feature,n_semantic,n_latent)
	sess = tf.Session()
	main_feed_dict = {model.image_input:train_X,model.label_input:train_labels,model.semantic_input:train_Y}
	test_feed_dict = {model.image_input:test_X,model.label_input:test_labels,model.semantic_input:test_Y}
	sess.run(tf.global_variables_initializer())
	saver = tf.train.Saver()
	epoch = 150
	batch_size = 64
	train_acc_list = []
	test_acc_list = []
	epoch_list = []
	for i in range(epoch):
		iteration = int(n_data/batch_size + 1)
		random_index = np.random.permutation(n_data)
		for j in range(iteration):
			if not j == iteration - 1:
				index = random_index[j*batch_size:(j+1)*batch_size]
			else:
				index = random_index[j*batch_size:]
			train_X_batch = train_X[index,:]
			train_labels_batch = train_labels[index]
			feed_dict = {model.image_input:train_X_batch,
			model.label_input:train_labels_batch,model.semantic_input:train_Y}
			sess.run(model.learner,feed_dict=feed_dict)
		loss = sess.run(model.final_loss,feed_dict=main_feed_dict)
		predict_label = sess.run(model.predict_label,feed_dict=main_feed_dict)
		train_acc = 100.0*sum(predict_label==train_labels)/float(train_labels.shape[0])
		predict_label = sess.run(model.predict_label,feed_dict=test_feed_dict)
		test_acc = 100.0*sum(predict_label==test_labels)/float(test_labels.shape[0])
		print('epoch '+str(i)+':train_loss:'+str(loss)+' train_acc:'+str(train_acc)+' test_acc:'+str(test_acc))
		if i%200 == 0:
			saver.save(sess,'./weight/model_'+str(i))
		train_acc_list.append(train_acc)
		test_acc_list.append(test_acc)
		epoch_list.append(i)

	
