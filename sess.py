import tensorflow as tf

a = tf.constant(5, name="input_a")
b = tf.constant(9, name="input_b")
c = tf.multiply(a, b, name="mul_c")

with tf.Session() as sess:
    print(sess.run(c))
