# name_scopes.py
import tensorflow as tf

with tf.Graph().as_default():
 ns1 = tf.constant(9,dtype=tf.float64,name='ns')
with tf.name_scope("prefix_name"):
 ns2 = tf.constant(9,dtype=tf.int32,name='ns')
 ns3 = tf.constant(9,dtype=tf.float64,name='ns')
print(ns1.name)
print(ns2.name)
print(ns3.name)