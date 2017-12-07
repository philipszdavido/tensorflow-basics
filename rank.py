# rank.py
import tensorflow as tf
# tensor object
t = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [2, 3, 4]]
# initialize session here
sess = tf.Session()

print("Tensor  Object: {}".format(t)) # prints [[1, 2, 3], [4, 5, 6], [7, 8, 9], [2, 3, 4]]
print("Rank: {}".format(sess.run(tf.rank(t)))) # prints the rank 2
print("Shape: {}".format(sess.run(tf.shape(t)))) # prints the shape [4, 3]
print("Size: {}".format(sess.run(tf.size(t)))) # prints the size 12

