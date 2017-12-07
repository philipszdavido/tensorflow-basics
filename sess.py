import tensorflow as tf
import numpy as np

a = tf.constant(5, name="input_a")
b = tf.constant(9, name="input_b")
c = tf.multiply(a, b, name="mul_c")

with tf.Session() as sess:
    print(sess.run(c))
init_val = tf.random_uniform((2,5),1, 4)
init_va = tf.random_normal([3,5],0,1)
var = tf.Variable(init_val, name='var')
print("pre run: \n{}".format(var))
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    post_var = sess.run(var)
print("\npost run: \n{}".format(post_var))

x_data = np.random.randn(5,10)
w_data = np.random.randn(10,1)
with tf.Graph().as_default():
    x = tf.placeholder(tf.float32,shape=(5,10))
    w = tf.placeholder(tf.float32,shape=(10,1))
    b = tf.fill((5,1),-1.)
    xw = tf.matmul(x,w)
    
    xwb = xw + b
    s = tf.reduce_max(xwb)
    with tf.Session() as sess:
        outs = sess.run(s,feed_dict={x: x_data,w: w_data})
print("outs = {}".format(outs))

