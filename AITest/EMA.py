import tensorflow as tf

v1 = tf.Variable(0, dtype = tf.float32)
step = tf.Variable(0, trainable = False)
beta = 0.99
ema = tf.train.ExponentialMovingAverage(beta, step)
maintain_averages_op = ema.apply([v1])

with tf.Session() as sess:
    init_op = tf.initialize_all_variables()
    sess.run(init_op)

    print(sess.run([v1, ema.average(v1)]))

    sess.run(tf.assign(v1, 5))
    sess.run(maintain_averages_op)

    print(sess.run([v1, ema.average(v1)]))