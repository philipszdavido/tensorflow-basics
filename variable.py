# variable.py
import tensorflow as tf

init_val = tf.random_normal((2,4),1,4)
var = tf.Variable(init_val, name='var')
print("pre run: \n{}".format(var))

init = tf.global_variables_initializer()

with tf.Session() as sess:
 sess.run(init)
 post_var = sess.run(var)

print("\npost run: \n{}".format(post_var))
