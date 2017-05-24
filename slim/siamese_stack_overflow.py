
#link to the stack: http://stackoverflow.com/questions/41172500/how-to-implement-metrics-learning-using-siamese-neural-network-in-tensorflow

# In[1]:

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.examples.tutorials.mnist import input_data
from math import sqrt
import numpy as np
from sklearn.manifold import TSNE
get_ipython().magic('matplotlib inline')
get_ipython().magic('pylab inline')


# In[2]:

mnist = input_data.read_data_sets('MNIST_data', one_hot=False)


# In[3]:

learning_rate = 0.00001
training_epochs = 15
batch_size = 100
display_step = 1
logs_path = './tensorflow_logs/mnist_metrics'
# Network Parameters
n_hidden_1 = 256 # 1st layer number of features
n_hidden_2 = 256 # 2nd layer number of features
n_input = 28*28 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)
margin = 1.0


# In[4]:

x_left = tf.placeholder(tf.float32, shape=[None, n_input], name='InputDataLeft')
x_right = tf.placeholder(tf.float32, shape=[None, n_input], name='InputDataRight')
label = tf.placeholder(tf.float32, shape=[None, 1], name='LabelData') # 0 if the same, 1 is different

x_image_left = x_left
x_image_right = x_right


# In[5]:

# def NN(inputs):

# In[6]:

def tfNN(x, weights, biases):
	x = tf.scalar_mul(1.0/256.0, x)
	layer_1 = tf.add(tf.matmul(x, weights['w1']), biases['b1'])
	layer_1 = tf.nn.relu(layer_1)
	layer_2 = tf.add(tf.matmul(layer_1, weights['w2']), biases['b2'])
	layer_2 = tf.nn.relu(layer_2)
	layer_3 = tf.add(tf.matmul(layer_2, weights['w3']), biases['b3'])
	out_layer = tf.add(tf.matmul(layer_3, weights['w4']), biases['b4'])
	return out_layer


# In[7]:

# Store layers weight & bias
weights = {
'w1': tf.Variable(tf.random_uniform([n_input, n_hidden_1], minval=-4*np.sqrt(6.0/(n_input + n_hidden_1)), maxval=4*np.sqrt(6.0/(n_input + n_hidden_1))), name='W1'),
'w2': tf.Variable(tf.random_uniform([n_hidden_1, n_hidden_2], minval=-4*np.sqrt(6.0/(n_hidden_1 + n_hidden_2)), maxval=4*np.sqrt(6.0/(n_hidden_1 + n_hidden_2))), name='W2'),
'w3': tf.Variable(tf.random_uniform([n_hidden_2, n_classes], minval=-4*np.sqrt(6.0/(n_hidden_2 + n_classes)), maxval=4*np.sqrt(6.0/(n_hidden_2 + n_classes))), name='W3'),
'w4': tf.Variable(tf.random_uniform([n_classes, 2], minval=-4*np.sqrt(6.0/(n_classes + 2)), maxval=4*np.sqrt(6.0/(n_classes + 2))), name='W4')
}
biases = {
'b1': tf.Variable(tf.truncated_normal([n_hidden_1]) / sqrt(n_hidden_1), name='b1'),
'b2': tf.Variable(tf.truncated_normal([n_hidden_2]) / sqrt(n_hidden_2), name='b2'),
'b3': tf.Variable(tf.truncated_normal([n_classes]) / sqrt(n_classes), name='b3'),
'b4': tf.Variable(tf.truncated_normal([2]) / sqrt(2), name='b4')
}


# In[8]:

with tf.name_scope('Model'):
	# Model
	pred_left = tfNN(x_image_left, weights, biases)
	pred_right = tfNN(x_image_right, weights, biases)
with tf.name_scope('Loss'):
	# Minimize error using cross entropy
	# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
	d = tf.reduce_sum(tf.square(pred_left - pred_right), 1)
	d_sqrt = tf.sqrt(d)
	loss = label * tf.square(tf.maximum(0.0, margin - d_sqrt)) + (1 - label) * d
	loss = 0.5 * tf.reduce_mean(loss)

with tf.name_scope('AdamOptimizer'):
	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)


# In[9]:

# Initializing the variables
init = tf.global_variables_initializer()
# Create a summary to monitor cost tensor
tf.scalar_summary("loss", loss)

# Merge all summaries into a single op
merged_summary_op = tf.merge_all_summaries()


# In[10]:

# Launch the graph
sess = tf.Session()
sess.run(init)
# op to write logs to Tensorboard
summary_writer = tf.train.SummaryWriter(logs_path, graph=tf.get_default_graph())

# Training cycle
for epoch in range(training_epochs):
	avg_loss = 0.0
	total_batch = int(mnist.train.num_examples / batch_size)
# Loop over all batches
for i in range(total_batch):
    left_batch_xs, left_batch_ys = mnist.train.next_batch(batch_size)
    right_batch_xs, right_batch_ys = mnist.train.next_batch(batch_size)
    labels = np.zeros((batch_size, 1))
    for l in range(batch_size):
        if left_batch_ys[l] == right_batch_ys[l]:
            labels[l, 0] = 0.0
        else:
            labels[l, 0] = 1.0
    _, l, summary = sess.run([optimizer, loss, merged_summary_op],
                             feed_dict = {
                                          x_left: left_batch_xs,
                                          x_right: right_batch_xs,
                                          label: labels,
                                         })
    # Write logs at every iteration
    summary_writer.add_summary(summary, epoch * total_batch + i)
    # Compute average loss
    avg_loss += l / total_batch
# Display logs per epoch step
if (epoch+1) % display_step == 0:
    print ("Epoch:", '%04d' % (epoch+1), "loss =", "{:.9f}".format(avg_loss))

print ("Optimization Finished!")

print ("Run the command line:\n"       "--> tensorboard --logdir=./tensorflow_logs "       "\nThen open http://0.0.0.0:6006/ into your web browser")


# In[11]:

# Test model
# Calculate accuracy
test_xs, test_ys = mnist.train.next_batch(5000)
ans = sess.run([pred_left], feed_dict = { x_left: test_xs})

# In[12]:

ans = ans[0]


# In[13]:

#test_ys


# In[14]:

figure(figsize=(10,10))
# scatter(r[:,0], r[:,1], c=[test_ys[x,:].argmax() for x in range(len(test_ys))])
scatter(ans[:,0], ans[:,1], c=test_ys[:])