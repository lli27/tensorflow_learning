import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/data/mnist_dataset", one_hot=True)

learning_rate = 0.001
training_iters = 2000
batch_size = 128
display_step = 10

n_input =784
n_classes = 10
dropout = 0.75
x = tf.placeholder(tf.float32,[None, n_input])
y = tf.placeholder(tf.float32,[None, n_classes])
keep_prob = tf.placeholder(tf.float32)

def conv2d(x, W, b, strides=1):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x,b)
    return tf.nn.relu(x)

def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

def conv_net(x, weights, biases, dropout):
    x = tf.reshape(x, shape=[-1, 28, 28, 1])
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    conv1 = maxpool2d(conv1, k=2)
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    conv2 = maxpool2d(conv2, k=2)
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    fc1 = tf.nn.dropout(fc1, dropout)
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out

weights = {'wc1':tf.Variable(tf.random_normal([5,5,1,32])), 
           'wc2':tf.Variable(tf.random_normal([5,5,32,64])),
           'wd1':tf.Variable(tf.random_normal([7*7*64,1024])),
           'out':tf.Variable(tf.random_normal([1024,n_classes]))}

biases = {'bc1':tf.Variable(tf.random_normal([32])),
           'bc2':tf.Variable(tf.random_normal([64])),
           'bd1':tf.Variable(tf.random_normal([1024])),
           'out':tf.Variable(tf.random_normal([n_classes]))}

pred = conv_net(x, weights, biases, keep_prob)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

correct_pred = tf.equal(tf.arg_max(pred,1),tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    step = 1
    while step * batch_size < training_iters:
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        test = batch_x[0]
        fig = plt.figure()
        plt.imshow(test.reshape((28,28),order='C'), cmap='Greys', interpolation='nearest')
        print(weights['wc1'].eval()[0])
        plt.imshow(weights['wc1'].eval()[0][0].reshape(4,8),cmap='Greys',interpolation='nearest')
        sess.run(optimizer, feed_dict={x:batch_x,y:batch_y,keep_prob:dropout})

        if step % display_step == 0:
            loss, acc = sess.run([cost, accuracy], feed_dict={x:batch_x, y:batch_y, keep_prob:1.0})
            print("Iter "+ str(step*batch_size) + ", Minibatch Loss= " + "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(acc))

        step += 1
    print("Optimization Finished!")
    print("Testing Accuracy: ", sess.run(accuracy, feed_dict={x:mnist.test.images[:256],y:mnist.test.labels[:256],keep_prob:1.0}))
