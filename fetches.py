# fetches.py
import tensorflow as tf

a = tf.add(7,9)
b = tf.multiply(a,9)

with tf.Session() as sess:
 fetches = [a,b]
 outs = sess.run(fetches)
print("outs = {}".format(outs))
print(type(outs[0]))
