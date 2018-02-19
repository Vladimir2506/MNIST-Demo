# Model.py
# Implement a CNN for MNIST
# Conv1 -> MaxPool1 -> Conv2 -> MaxPool2 -> FC1 -> FC2
# ReLU ---> Softmax

import tensorflow as tf
import PrepareData as pd

# sess = tf.InteractiveSession()

# Constant of MNIST
INPUT_NODE = 784
OUTPUT_NODE = 10

# Hyperparametres of NN
EPOCHS_MAX = 2000
BATCH_SIZE = 64        # size of mini-batch
LEARNING_RATE = 0.001   # learning rate for Adam
BETA1 = 0.9             # Beta1 for Adam
BETA2 = 0.999           # Beta2 for Adam
EPSILON = 1e-8          # Epsilon for Adam

# Initialiser for paremetres
def InitWeight(shape):
    initialRandom = tf.truncated_normal(shape, stddev = 0.1)    # Random initialise weight matrix with a standard deviation of 0.1
    return tf.Variable(initialRandom)

def InitBias(shape):
    initialZero = tf.constant(0.0, shape = shape)     # Zero initialise bias matrix
    return tf.Variable(initialZero)

# Layers for CNN
def ConvLayer(inputMat, kernelMat):
    return tf.nn.conv2d(inputMat, kernelMat, strides = [1, 1, 1, 1], padding = 'SAME')

def MaxPoolLayer(inputMat):
    return tf.nn.max_pool(inputMat, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')

# Architecture of Model

# Init parametres
x = tf.placeholder(dtype = tf.float32, shape = [None, INPUT_NODE])
y = tf.placeholder(dtype = tf.float32, shape = [None, OUTPUT_NODE])

weightConv1 = InitWeight([5, 5, 1, 32])
biasConv1 = InitBias([32])

weightConv2 = InitWeight([5, 5, 32, 64])
biasConv2 = InitBias([64])

weightFC1 = InitWeight([7 * 7 * 64, 1024])
biasFC1 = InitBias([1024])

weightFC2 = InitWeight([1024, 10])
biasFC2 = InitBias([10])

keepProb = tf.placeholder(dtype = tf.float32)

# Define Computation Graph
inputNodes = tf.reshape(x, [-1, 28, 28, 1])

linearConv1 = ConvLayer(inputNodes, weightConv1) + biasConv1
activationConv1 = tf.nn.relu(linearConv1)
poolConv1 = MaxPoolLayer(activationConv1)

linearConv2 = ConvLayer(poolConv1, weightConv2) + biasConv2
activationConv2 = tf.nn.relu(linearConv2)
poolConv2 = MaxPoolLayer(activationConv2)

features = tf.reshape(poolConv2, [-1, 7 * 7 * 64])
linearFC1 = tf.matmul(features, weightFC1) + biasFC1
activationFC1 = tf.nn.relu(linearFC1)
dropoutFC1 = tf.nn.dropout(activationFC1, keepProb)

linearFC2 = tf.matmul(dropoutFC1, weightFC2) + biasFC2
activationFC2 = tf.nn.softmax(linearFC2)

# Define training and prediction
outputNodes = activationFC2
groundTruth = y
crossEntropyLoss = - tf.reduce_sum(groundTruth * tf.log(outputNodes), [1])
crossEntropyCost = tf.reduce_mean(crossEntropyLoss)
trainStep = tf.train.AdamOptimizer(learning_rate = LEARNING_RATE, beta1 = BETA1, beta2 = BETA2, epsilon = EPSILON).minimize(crossEntropyCost)
prediction = tf.equal(tf.argmax(outputNodes, 1), tf.argmax(groundTruth, 1))
accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))

# Run
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    mnist = pd.LoadDataset()
    for epoch in range(EPOCHS_MAX):
        batch = mnist.train.next_batch(BATCH_SIZE)
        if epoch % 100 == 0:
            trainAcc = accuracy.eval(feed_dict = {x:batch[0], y:batch[1], keepProb:1})
            print("Epoch[%d] = %g" %(epoch, trainAcc))
        sess.run(trainStep, feed_dict = {x:batch[0], y:batch[1], keepProb:0.5})
    
    print("Cross Validation : %g" %accuracy.eval(feed_dict={x:mnist.test.images, y:mnist.test.labels, keepProb:1.0}))


