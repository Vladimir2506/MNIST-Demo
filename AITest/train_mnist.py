# train_mnist.py
# Implement a CNN for MNIST
# Conv1 -> MaxPool1 -> Conv2 -> MaxPool2 -> FC1 -> FC2
# ReLU ---> Softmax

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data

# Prepare MNIST
def load_data():
    dataset_mnist = input_data.read_data_sets('F:\\MNIST', one_hot = True)
    return dataset_mnist

# Constant of MNIST
INPUT_NODE = 784
OUTPUT_NODE = 10

# Hyperparametres of NN
EPOCHS_MAX = 10         # Epoch of training
BATCH_SIZE = 512         # Size of mini-batch
LEARNING_RATE = 0.001   # Learning rate for Adam
BETA1 = 0.9             # Beta1 for Adam
BETA2 = 0.999           # Beta2 for Adam
EPSILON = 1e-8          # Epsilon for Adam

# Initialiser for paremetres
def init_weight(shape):

    # Random initialise weight matrix with a standard deviation of 0.1
    
    initialRandom = tf.truncated_normal(shape, stddev = 0.1)    
    
    return tf.Variable(initialRandom)

def init_bias(shape):
    
    # Zero initialise bias matrix

    initialZero = tf.constant(0.0, shape = shape)     
    
    return tf.Variable(initialZero)

# Layers for CNN
def conv_layer(inputMat, kernelMat):

    return tf.nn.conv2d(inputMat, kernelMat, 
                        strides = [1, 1, 1, 1], 
                        padding = 'SAME')

def maxpool_layer(inputMat):

    return tf.nn.max_pool(inputMat, 
                          ksize = [1, 2, 2, 1], 
                          strides = [1, 2, 2, 1], 
                          padding = 'SAME')

# Architecture of Model

# Init parametres
x = tf.placeholder(dtype = tf.float32, shape = [None, INPUT_NODE])
y = tf.placeholder(dtype = tf.float32, shape = [None, OUTPUT_NODE])

weight_conv_1 = init_weight([5, 5, 1, 32])
bias_conv_1 = init_bias([32])

weight_conv_2 = init_weight([5, 5, 32, 64])
bias_conv_2 = init_bias([64])

weight_fc_1 = init_weight([7 * 7 * 64, 1024])
bias_fc_1 = init_bias([1024])

weight_fc_2 = init_weight([1024, 10])
bias_fc_2 = init_bias([10])

keep_prob = tf.placeholder(dtype = tf.float32)

# Define Computation Graph
input_nodes = tf.reshape(x, [-1, 28, 28, 1])

linear_conv_1 = conv_layer(input_nodes, weight_conv_1) + bias_conv_1
activation_conv_1 = tf.nn.relu(linear_conv_1)
pool_conv_1 = maxpool_layer(activation_conv_1)

linear_conv_2 = conv_layer(pool_conv_1, weight_conv_2) + bias_conv_2
activation_conv_2 = tf.nn.relu(linear_conv_2)
pool_conv_2 = maxpool_layer(activation_conv_2)

features = tf.reshape(pool_conv_2, [-1, 7 * 7 * 64])
linear_fc_1 = tf.matmul(features, weight_fc_1) + bias_fc_1
activation_fc_1 = tf.nn.relu(linear_fc_1)
dropout_fc_1 = tf.nn.dropout(activation_fc_1, keep_prob)

linear_fc_2 = tf.matmul(dropout_fc_1, weight_fc_2) + bias_fc_2
activation_fc_2 = tf.nn.softmax(linear_fc_2)

# Define training, loss and prediction
output_nodes = activation_fc_2
ground_truth = y

cross_entropy_loss = - tf.reduce_sum(ground_truth * tf.log(output_nodes), [1])
cross_entropy_loss = tf.reduce_mean(cross_entropy_loss)

train_step = tf.train.AdamOptimizer(learning_rate = LEARNING_RATE,
                                    beta1 = BETA1,
                                    beta2 = BETA2, 
                                    epsilon = EPSILON).minimize(cross_entropy_loss)

prediction = tf.equal(tf.argmax(output_nodes, 1), tf.argmax(ground_truth, 1))
accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))

# Run 
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config = config) as sess:
    
    sess.run(tf.global_variables_initializer())
    
    mnist = load_data()

    # Train
    train_data = mnist.train
    completed = 0
    
    # Show accuracy every epoch
    while train_data.epochs_completed <= EPOCHS_MAX:
        
        batch = train_data.next_batch(BATCH_SIZE)
        
        if train_data.epochs_completed > completed and train_data.epochs_completed <= EPOCHS_MAX:
            
            completed = train_data.epochs_completed
            train_accuracy = accuracy.eval(feed_dict = 
                                           {x:batch[0], y:batch[1], keep_prob:1.0})
            print('Epoch[%d] Train Accuracy = %g' 
                  %(train_data.epochs_completed, train_accuracy))

        if train_data.epochs_completed < EPOCHS_MAX:
            sess.run(train_step, feed_dict = 
                     {x:batch[0], y:batch[1], keep_prob:0.5})
    
    # Do cross validation
    print('Cross Validation Accuracy = %g' 
          %accuracy.eval(feed_dict = 
                         {x:mnist.validation.images, y:mnist.validation.labels, keep_prob:1.0}))
    
    # Performance test
    print('Test for 10 digits:')
    test_data = mnist.test
    Xs = test_data.images
    Ys = test_data.labels
    Ps = output_nodes.eval(feed_dict = 
                           {x:Xs, y:Ys, keep_prob:1.0}).argmax(axis = 1)
    Size = test_data.num_examples
    
    for i in range(0, 10):
        ind = np.random.randint(Size)
        label = Ys[ind].argmax(axis = 0)
        image = Xs[ind].reshape([28, 28])
        pred = Ps[ind]
        plt.title('Example: %d Label: %d Prediction:%d' % (ind, label, pred))
        plt.imshow(image, cmap=plt.get_cmap('gray_r'))
        plt.show()


