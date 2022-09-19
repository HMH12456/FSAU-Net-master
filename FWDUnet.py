import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

def conv2D(input, filters, kernel_size=3, stride=1, padding='SAME', d_rate=1):
    return tf.layers.conv2d(input, filters=filters, kernel_size=kernel_size,
                            padding=padding, dilation_rate=d_rate, strides=stride,
                            kernel_initializer=tf.variance_scaling_initializer())


def bn(input, is_training=True):
    return tf.layers.batch_normalization(inputs=input,training=is_training,
                                      center=True,scale=True,fused=True)

def myattention(input,is_training,reta):
    Avgput = tf.layers.average_pooling2d(input,2,2)
    Avgput = conv2D(Avgput,Avgput.shape[3]*reta,3)
    Avgput = tf.nn.relu(bn(Avgput,is_training))
    Avgput = conv2D(Avgput, Avgput.shape[3] // reta,3) # ----尺寸 256 x 256
    Avgput = tf.nn.relu(bn(Avgput,is_training))
    output = tf.image.resize_images(Avgput,tf.shape(input)[1:3])
    output = tf.nn.sigmoid(output) * input
    return output

def spatial_attention(input_feature, name):
    kernel_size = 7
    kernel_initializer = tf.contrib.layers.variance_scaling_initializer()
    with tf.variable_scope(name):
        avg_pool = tf.reduce_mean(input_feature, axis=[3], keepdims=True)
        assert avg_pool.get_shape()[-1] == 1
        max_pool = tf.reduce_max(input_feature, axis=[3], keepdims=True)
        assert max_pool.get_shape()[-1] == 1
        concat = tf.concat([avg_pool, max_pool], 3)
        assert concat.get_shape()[-1] == 2

        concat = tf.layers.conv2d(concat,
                                  filters=1,
                                  kernel_size=[kernel_size, kernel_size],
                                  strides=[1, 1],
                                  padding="same",
                                  activation=None,
                                  kernel_initializer=kernel_initializer,
                                  use_bias=False,
                                  name='conv')
        assert concat.get_shape()[-1] == 1
        concat = tf.sigmoid(concat, 'sigmoid')

    return input_feature * concat

def FWDUnet(input,is_training=True):
    conv1 = conv2D(input, 32)
    for i in range(32):
        plt.matshow(conv1[0,:,:,i], cmap='viridis')
        plt.axis('off')
        plt.savefig('features//feature-{}.png'.format(i), bbox_inches='tight')
    plt.show(A)
    bn1 = tf.nn.relu(bn(conv1, is_training))
    attention_output = myattention(bn1,is_training,2)
    # bn1_reta = channel_attention(bn1,"myunet",8)   ##从这里开始是我自己的部分
    conv1_1 = conv2D(attention_output, 32)
    bn1_1 = tf.nn.relu(bn(conv1_1, is_training))
    pool1 = tf.layers.max_pooling2d(bn1_1, 2, 2)



    conv2 = conv2D(pool1, 64)
    bn2 = tf.nn.relu(bn(conv2, is_training))
    attention_output1 = myattention(bn2,is_training,2) #这个参数需要不断做实验来找出来
    conv2_1 = conv2D(attention_output1, 64)
    bn2_1 = tf.nn.relu(bn(conv2_1, is_training))
    pool2 = tf.layers.max_pooling2d(bn2_1, 2, 2)

    conv3 = conv2D(pool2, 128)
    bn3 = tf.nn.relu(bn(conv3, is_training))
    attention_output3 = myattention(bn3,is_training,2)
    conv3_1 = conv2D(attention_output3, 128)
    bn3_1 = tf.nn.relu(bn(conv3_1, is_training))
    pool3 = tf.layers.max_pooling2d(bn3_1, 2, 2)

    conv4 = conv2D(pool3, 256)
    bn4 = tf.nn.relu(bn(conv4, is_training))
    attention_output4 = myattention(bn4,is_training,2)
    conv4_1 = conv2D(attention_output4, 256)
    bn4_1 = tf.nn.relu(bn(conv4_1, is_training))
    pool4 = tf.layers.max_pooling2d(bn4_1, 2, 2)

    conv5 = conv2D(pool4, 512)
    bn5 = tf.nn.relu(bn(conv5, is_training))
    conv5_1 = conv2D(bn5, 512)
    bn5_1 = tf.nn.relu(bn(conv5_1, is_training))

    up1 = tf.image.resize_images(bn5_1, tf.shape(bn5_1)[1:3] * 2)
    up1 = conv2D(up1, 256)
    up1 = tf.concat([up1, bn4_1], axis=3)
    up1 = conv2D(up1, 256)
    up1 = tf.nn.relu(bn(up1, is_training))
    up1 = spatial_attention(up1,"spatial_attention1")
    up1 = conv2D(up1, 256)
    up1 = tf.nn.relu(bn(up1, is_training))

    up2 = tf.image.resize_images(up1, tf.shape(up1)[1:3] * 2)
    up2 = conv2D(up2, 128)
    up2 = tf.concat([up2, bn3_1], axis=3)
    up2 = conv2D(up2, 128)
    up2 = tf.nn.relu(bn(up2, is_training))
    up2 = spatial_attention(up2,"spatial_attention2")
    up2 = conv2D(up2, 128)
    up2 = tf.nn.relu(bn(up2, is_training))

    up3 = tf.image.resize_images(up2, tf.shape(up2)[1:3] * 2)
    up3 = conv2D(up3, 64)
    up3 = tf.concat([up3, bn2_1], axis=3)
    up3 = conv2D(up3, 64)
    up3 = tf.nn.relu(bn(up3, is_training))
    up3 = spatial_attention(up3,"spatial_attention3")
    up3 = conv2D(up3, 64)
    up3 = tf.nn.relu(bn(up3, is_training))

    up4 = tf.image.resize_images(up3, tf.shape(up3)[1:3] * 2)
    up4 = conv2D(up4, 32)
    up4 = tf.concat([up4, bn1_1], axis=3)
    up4 = conv2D(up4, 32)
    up4 = tf.nn.relu(bn(up4, is_training))
    up4 = spatial_attention(up4,"spatial_attention4")
    up4 = conv2D(up4, 32)
    up4 = tf.nn.relu(bn(up4, is_training))

    final = conv2D(up4, 1, 1)

    # fig4, ax4 = plt.subplots(nrows=1, ncols=16, figsize=(16, 1))
    # for i in range(16):
    #     ax4[i].imshow(final[i][0])
    #
    # plt.title('Pool1 16x14x14')
    # plt.show()


    return final


