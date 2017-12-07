# feed_dict.py

import tensorflow as tf

# Create Operations, Tensors, etc (using the default graph)
a = tf.add(5, 9)
b = tf.multiply(a, 7)
# Start up a `Session` using the default graph
sess = tf.Session()
# Define a dictionary that says to replace the value of `a` with 45
replace_dict = {a: 45}
# Run the session, passing in `replace_dict` as the value to `feed_dict`
print(sess.run(b, feed_dict=replace_dict)) # returns 315
# Close the graph, release its resources
sess.close()

