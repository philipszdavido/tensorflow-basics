# placeholder.py
import tensorflow as tf

x = tf.placeholder("float", None)
y = x * 5

with tf.Session() as sess:
    ans = sess.run(y, feed_dict={x: [2, 4, 6]})
    print(ans) 
