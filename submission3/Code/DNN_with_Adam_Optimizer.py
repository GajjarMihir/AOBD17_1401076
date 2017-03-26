
# mnist Data set:
# 60,000 training examples from handwritted digits.
# Anything from a zero to a nine and they are all 28*28 pixels and
# hence there will be 784 total pixels. 10,000 testing examples.
# Train with the 60,000 and test with the 10,000 examples. 
# The 10,000 samples are the unique, they are not the stuff that we trained on.
# They are black and white thresholded images.

# Features - pixels - 0 or 1. White or Black.

'''
input > weight > hidden layer 1 (activation function) > weights > hidden layer 2
(activation function) > weights > output layer.
# Feed Forward Neural network.
compare output to intended output > cost or loss function (cross entropy) - how wrong are we? how (not) close to our intended target are we ?
optimization function (optimizer) > minimize cost (AdamOptimizer....SGD, AdaGrad)

backpropagation

feed forward + backprop = epoch (one cycle of feed forward and backprop)
After 10,15,20 times, after so long each time we cycle through we are lowering our cost function. Initially the cost will be very high and as the time goes it kinda drops down and levels out/or some wiggle (the cost might go up). Then we are not making any more progress.

'''

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)
# 10 classes, 0-9
'''
0 = 0
1 = 1
2 = 2

What one_hot does is
0 = [1,0,0,0,0,0,0,0,0,0]
1 = [0,1,0,0,0,0,0,0,0,0]
2 = [0,0,1,0,0,0,0,0,0,0]
3 = [0,0,0,1,0,0,0,0,0,0]

one_hot - one element is hot(on) and the rest are cold or off.

'''

# The number of nodes need not be identical. Depends on the thing you are trying to model.
n_nodes_hl1 = 1000
n_nodes_hl2 = 1000
n_nodes_hl3 = 1000
n_nodes_hl4 = 1000

n_classes = 10

# Goes through the batches of 100 of features and feeds them through our network at a time and manipulates the weights and then do another batch and manipulate the weights a little bit by batches of 100 images.
batch_size = 25

#Defining a couple place-holdering variables.


# height x width
# x is the data and y is the label of that data.
x = tf.placeholder('float',[None,784]) # 784 pixels wide.
y = tf.placeholder('float')

# data is the raw input.
def neural_network_model(data):

	# Bias is for the case when all the input data is zero. 

	hidden_1_layer = {'weights':tf.Variable(tf.random_normal([784,n_nodes_hl1])),
					  'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}

	hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1,n_nodes_hl2])),
					  'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}

	hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2,n_nodes_hl3])),
					  'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}

	hidden_4_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3,n_nodes_hl4])),
					  'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}

	output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl4,n_classes])),
					  'biases':tf.Variable(tf.random_normal([n_classes]))}


	# (input_data * weights) + biases 	

	l1 = tf.add(tf.matmul(data,hidden_1_layer['weights']), hidden_1_layer['biases'])	
	l1 = tf.nn.relu(l1)		  
	# relu rectify linear that is your activation function. Rectified linear is like you threshold function.	

	l2 = tf.add(tf.matmul(l1,hidden_2_layer['weights']),hidden_2_layer['biases'])	
	l2 = tf.nn.relu(l2)	

	l3 = tf.add(tf.matmul(l2,hidden_3_layer['weights']), hidden_3_layer['biases'])	
	l3 = tf.nn.relu(l3)

	l4 = tf.add(tf.matmul(l3,hidden_4_layer['weights']), hidden_4_layer['biases'])	
	l4 = tf.nn.relu(l4)

	output = tf.matmul(l4,output_layer['weights']) + output_layer['biases']

	return output

	# At this point we now have modeled a neural network.
	# We are done with setting up of the computation graph.

# x is the input data.
def train_neural_network(x):
	prediction = neural_network_model(x)
	# Calculates the difference between the prediction that we got to the known label that we have and both of these are in one hot format and hence we have done one hot = true. Hence the output layer is of that shape.
	# We can have out output be of whatever shape. The output basically will just always be the shape of the testing sets labels. Training and testing sets.
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = prediction,labels = y))

	# now ideally we want no difference between the predicted and the original data.
	# Hence we need to minimize the cost.
	# It is similar to stochastic gradient descent.
	# Adam optimizer does have a paremeter called the learning rate. By Default: Learning rate = 0.001. Hence we are not modifying that.

	optimizer = tf.train.AdamOptimizer(learning_rate=0.007).minimize(cost)

	# cycles feed forward + backprop
	hm_epochs = 46

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		# Training the network
		for epoch in range(hm_epochs):
			epoch_loss = 0
			# _ is a shorthand for variable that we dont care about.	
			for _ in range(int(mnist.train.num_examples/batch_size)):
				# Basically it chunks through the data set for you.
				epoch_x, epoch_y = mnist.train.next_batch(batch_size)
				_, c = sess.run([optimizer,cost], feed_dict = {x: epoch_x, y: epoch_y})
				epoch_loss += c
			print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)
	
		# tf.argmax returns the index of the maximum value in the respective arrays.
		# tells whether both of these are identical.
		correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
		accuracy = tf.reduce_mean(tf.cast(correct,'float'))
		print('Accuracy:',accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))
		# We compare the prediction to the actual label.
		


train_neural_network(x)




					  			  



