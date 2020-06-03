import tensorflow as tf
slim = tf.contrib.slim

# standard convolution layer
def conv2d(x, inputFeatures, outputFeatures, name, train=True,bias=True):
    with tf.variable_scope(name):
        w = tf.get_variable("w",[5,5,inputFeatures, outputFeatures], initializer=tf.truncated_normal_initializer(stddev=0.02), trainable=train)
        b = tf.get_variable("b",[outputFeatures], initializer=tf.constant_initializer(0.0))
        if bias:
            conv = tf.nn.conv2d(x, w, strides=[1,2,2,1], padding="SAME") + b
        else:
            conv = tf.nn.conv2d(x, w, strides=[1,2,2,1], padding="SAME")
        return conv



def conv_transpose(x, outputShape, name):
    with tf.variable_scope(name):
        # h, w, out, in
        w = tf.get_variable("w",[5,5, outputShape[-1], x.get_shape()[-1]], initializer=tf.truncated_normal_initializer(stddev=0.02))
        b = tf.get_variable("b",[outputShape[-1]], initializer=tf.constant_initializer(0.0))
        convt = tf.nn.conv2d_transpose(x, w, output_shape=outputShape, strides=[1,2,2,1])
        return convt



# leaky reLu unit
def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)


# fully-conected layer
def dense(x, inputFeatures, outputFeatures, scope=None, with_w=False):
    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [inputFeatures, outputFeatures], tf.float32, tf.random_normal_initializer(stddev=0.02))
        bias = tf.get_variable("bias", [outputFeatures], initializer=tf.constant_initializer(0.0))
        if with_w:
            return tf.matmul(x, matrix) + bias, matrix, bias
        else:
            return tf.matmul(x, matrix) + bias

def decoder_lezama(z, out_shape, dim=20, reuse=False):
    with tf.variable_scope("decoder_lezama") as scope:
        if reuse:
                scope.reuse_variables()
        gf_dim=64
        batchsize = int(z.shape[0])
        h = int(out_shape[0])
        w = int(out_shape[1])
        in_channels = int(out_shape[2])
        df_dim=h

        s2, s4, s8, s16 = int(gf_dim/2), int(gf_dim/4), int(gf_dim/8), int(gf_dim/16)
        
        z0 = dense(z, dim, gf_dim*8*s16*s16, scope='g_h0_lin')
        
        #Deconv
        h0 = tf.contrib.layers.batch_norm(tf.reshape(z0, [-1, s16, s16, gf_dim * 8]))
        h0 = tf.nn.relu(h0)

        h1 = tf.contrib.layers.batch_norm(conv_transpose(h0, [batchsize, s8, s8, gf_dim*4], "g_h1"))
        h1 = tf.nn.relu(h1)
        
        h2 = tf.contrib.layers.batch_norm(conv_transpose(h1, [batchsize, s4, s4, gf_dim*2], "g_h2"))
        h2 = tf.nn.relu(h2)
        
        h3 = tf.contrib.layers.batch_norm(conv_transpose(h2, [batchsize, s2, s2, gf_dim*1], "g_h3"))
        h3 = tf.nn.relu(h3)
        
        h4 = tf.nn.sigmoid(conv_transpose(h3, [batchsize, h, w, in_channels], "g_h4"))
        
        return h4
        
def encoder_lezama(X, dim=20, reuse=False):
    with tf.variable_scope("encoder_lezama") as scope:
        if reuse:
                scope.reuse_variables()
        batchsize = int(X.shape[0])
        gf_dim=64
        df_dim=64
        h=64
        w=64
        c=int(X.shape[3])
        s2, s4, s8, s16 = int(gf_dim/2), int(gf_dim/4), int(gf_dim/8), int(gf_dim/16)
        
        #Conv
        h0 = lrelu(conv2d(X, c, df_dim, name='d_h0_conv'))
        
        h1 = tf.contrib.layers.batch_norm(conv2d(h0, df_dim, df_dim*2, name='d_h1_conv'))
        h1 = lrelu(h1) 
        
        h2 = tf.contrib.layers.batch_norm(conv2d(h1, df_dim*2, df_dim*4, name='d_h2_conv'))
        h2 = lrelu(h2) 
        
        h3 = tf.contrib.layers.batch_norm(conv2d(h2, df_dim*4, df_dim*8, name='d_h3_conv'))
        h3 = lrelu(h3)
        
        h3 = tf.reshape(h3, [batchsize, -1])

        return dense(h3, 8192, dim, scope='g_h0_lin')
