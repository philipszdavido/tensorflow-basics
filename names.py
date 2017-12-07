# names.py
import tensorflow as tf

with tf.Graph().as_default():
 five1 = tf.constant(5, dtype=tf.float64, name='five')
 five2 = tf.constant(5, dtype=tf.int32, name='five')
 six = tf.constant(6, dtype=tf.int32, name='six')
print(five1.name)
print(five2.name)
print(six.name)
