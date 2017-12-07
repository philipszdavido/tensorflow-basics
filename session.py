# session.py
import tensorflow as tf

f = tf.constant(5)

sess = tf.Session()
outs = sess.run(f)
print(outs)
