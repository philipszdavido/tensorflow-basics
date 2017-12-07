# tensor_shapes.py
import tensorflow as tf

# Shapes that specify a 0-D Tensor (scalar) 
# e.g. any single number: 7, 1, 3, 4, etc.
s_0_list = []
s_0_tuple = ()

list_0_D = tf.random_uniform(s_0_list, 1, 4)
tuple_0_D = tf.random_uniform(s_0_tuple, 1, 5)

list_0_D_var = tf.Variable(list_0_D, name='list_0')
tuple_0_D_var = tf.Variable(tuple_0_D, name='tuple_0')

# Shape that describes a vector of length 3
# e.g. [1, 2, 3]
s_1 = [3]
s_1_val = tf.random_uniform(s_1, 1, 4)
s_1_var = tf.Variable(s_1_val, name='s_1')

# Shape that describes a 3-by-2 matrix
# e.g [[1 ,2],
# [3, 4],
# [5, 6]]
s_2 = (3, 2)
s_2_val = tf.random_uniform(s_2, 1, 4)
s_2_var = tf.Variable(s_2_val, name='s_2')

init = tf.global_variables_initializer()
with tf.Session() as sess:
 sess.run(init)
 print("\nList 0-D Tensor: \n{}".format(sess.run(list_0_D_var)))
 print("\nTuple 0-D Tensor: \n{}".format(sess.run(tuple_0_D_var)))

 print("\n1-D Tensor: \n{}".format(sess.run(s_1_var)))

 print("\nMatrix 3-by-2: \n{}".format(sess.run(s_2_var)))
 print(sess.run(tf.shape(s_2_val, name="mystery_shape")))