import tensorflow as tf

x1 = tf.ones(shape=[336,9,9,32])

y1 = tf.layers.conv2d_transpose(x1, 1, 4, strides=2, activation=tf.nn.relu,padding='VALID')


# w=tf.ones([3,3,128,256])
# y2 = tf.nn.conv2d_transpose(x1, w, output_shape=[64,14,14,128], strides=[1,2,2,1], padding='SAME')  # x2,y2 inverse
# x2=tf.nn.conv2d(y2,w,strides=[1,2,2,1],padding='SAME')

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # y1_value,y2_value,x2_value=sess.run([y1,y2,x2])
    y1_value = sess.run(y1)

print(y1_value.shape)
# print(y2_value.shape)
# print(x2_value.shape)