

import tensorflow as tf
from tensorflow.keras.layers import Layer
import numpy as np


def _diagonal_initializer(shape, *ignored, **ignored_too):
    return tf.eye(shape[0], shape[1], dtype=tf.float32)


def _potts_model_initializer(shape, *ignored, **ignored_too):
    return -1 * _diagonal_initializer(shape)

def _get_ind(dz):
    if dz == 0:
        return 0, 0
    elif dz < 0:
        return 0, -dz
    elif dz > 0:
        return dz, 0
    else:
        return 0, 0

class ConvCRF(Layer):
    """
    Implement the ConvLayer described in:
    Convolutional CRFs for Semantic Segmentation
    Marvin T. T. Teichmann, Roberto Cipolla
    """
    def __init__(self, image_dims, filter_size,blur,num_classes,
                 theta_alpha, theta_beta, theta_gamma,
                 num_iterations, normalize=True, **kwargs):
        self.image_dims = image_dims
        self.filter_size=filter_size
        self.blur=blur
        self.num_classes = num_classes
        self.theta_alpha = theta_alpha
        self.theta_beta = theta_beta
        self.theta_gamma = theta_gamma
        self.num_iterations = num_iterations
        self.gauss_ker_weight_train = False
        self.spatial_ker_weights = None
        self.bilateral_ker_weights = None
        self.compatibility_matrix = None
        self.gauss_ker_weights=None
        self.normalize = normalize

        self._mesh = tf.cast(tf.stack(
            tf.meshgrid(tf.range(self.image_dims[0]), tf.range(self.image_dims[1]), indexing="ij"),
            0
        ), tf.float32) # (2, h, w)
        super(ConvCRF, self).__init__(**kwargs)

    def build(self, input_shape):
        # Weights of the spatial kernel
        self.spatial_ker_weights = self.add_weight(name='spatial_ker_weights',
                                                   shape=(self.num_classes, self.num_classes),
                                                   dtype=tf.float32,
                                                   initializer=_diagonal_initializer,
                                                   trainable=True)

        # Weights of the bilateral kernel
        self.bilateral_ker_weights = self.add_weight(name='bilateral_ker_weights',
                                                     shape=(self.num_classes, self.num_classes),
                                                     dtype=tf.float32,
                                                     initializer=_diagonal_initializer,
                                                     trainable=True)

        # Compatibility matrix
        self.compatibility_matrix = self.add_weight(name='compatibility_matrix',
                                                    shape=(self.num_classes, self.num_classes),
                                                    dtype=tf.float32,
                                                    initializer=_potts_model_initializer,
                                                    trainable=True)

        super(ConvCRF, self).build(input_shape)

    def pos_feature(self, bs):
        _mesh = tf.tile(tf.expand_dims(self._mesh, 0), (bs, 1, 1, 1)) #(b, 2, h, w)
        return 1.0/ self.theta_gamma * _mesh

    def color_pos_feature(self, rgb):
        # rgb (b, 3, h, w)
        bs = tf.shape(rgb)[0]
        _mesh = tf.tile(tf.expand_dims(self._mesh, 0), (bs, 1, 1, 1)) #(b, 2, h, w)
        rgb_norm = 1.0/ self.theta_alpha * rgb
        pos_norm = 1.0/ self.theta_beta * _mesh

        return tf.concat([rgb_norm,pos_norm],axis=1) #(b, 5, h, w)

    def _create_convolutional_filters(self, features):

        #features NCHW
        span=self.filter_size//2
        if self.blur>1:
            features=tf.keras.layers.AveragePooling2D((self.blur,self.blur),(self.blur,self.blur),padding='same',
                                                        data_format="channels_first")(features)

        bs = tf.shape(features)[0]
        c =  tf.shape(features)[1]
        h =  tf.shape(features)[2]
        w =  tf.shape(features)[3]

        gaussian_filter = tf.zeros([bs, self.filter_size, self.filter_size, h, w],dtype=tf.float32)
        indices = tf.stack(
            tf.meshgrid(
                tf.range(bs),
                tf.range(self.filter_size),
                tf.range(self.filter_size), 
                tf.range(h), 
                tf.range(w), 
                indexing='ij'
            ), -1) # expect b, f, f, h, w, 5

        for dx in range(-span,span+1):
            for dy in range(-span,span+1):
                
                dx1,dx2=_get_ind(dx)
                dy1,dy2=_get_ind(dy)
                
                # features are (b, c, h, w)
                feature_1 = tf.slice(features, (0, 0, dx1, dy1), (bs, c, h - dx2 - dx1, w - dy2 - dy1))
                feature_2 = tf.slice(features, (0, 0, dx2, dy2), (bs, c, h - dx1 - dx2, w - dy1 - dy2))

                diff_sq=(feature_1-feature_2)*(feature_1-feature_2)

                diff_exp=tf.expand_dims(tf.expand_dims(tf.exp(tf.reduce_sum(-0.5*diff_sq,axis=1)), 1), 1)
                idx = tf.slice(indices, (0, dx+span, dy+span, dx2, dy2, 0), (bs, 1, 1, h - dx1 - dx2, w - dy1 - dy2, 5))

                gaussian_filter = tf.tensor_scatter_nd_update(
                    gaussian_filter,
                    idx,
                    diff_exp,
                )
        
        return  tf.reshape(tf.expand_dims(gaussian_filter, 1), (bs, 1, -1, h, w))


    def compute_gaussian(self,input,filter):

        # expect b,c,h,w
        input = tf.transpose(input, (0,2,3,1))

        if self.blur>1:
            input = tf.keras.layers.AveragePooling2D(
                (self.blur,self.blur),
                (self.blur,self.blur),
                padding="SAME",
            )(input)

        bs, h, w, c = tf.shape(input)[0], tf.shape(input)[1], tf.shape(input)[2], tf.shape(input)[3]
        
        input_col = tf.image.extract_patches(
            input,
            [1,self.filter_size,self.filter_size,1],
            strides=[1,1,1,1],
            rates=[1,1,1,1],
            padding='SAME'
        )

        print(f"{input_col.shape=}")
        input_col = tf.reshape(input_col, (bs, -1, self.filter_size*self.filter_size, c))
        input_col = tf.transpose(input_col, (0, 3, 2, 1)) # (b, c, f*f, h*w)
        input_col = tf.reshape(input_col, (bs, c, -1, h, w))
        #(bs,c,self.filter_size*filter_size,h,w)
        # input_col = tf.reshape(input_col,(bs, h, w, self.filter_size*self.filter_size, c))
        # input_col = tf.transpose(input_col,perm=[0,4,3,1,2]) #(b, c, f*f, h, w)
        product = input_col * filter
        product = tf.reduce_sum(product,axis=2)

        return product

    @tf.function
    def call(self, input):
        
        unaries=tf.transpose(input[0],perm=(0,3,1,2)) # b,c,h,w
        rgb=tf.transpose(input[1],perm=(0,3,1,2))

        bs, c, h, w = (
            tf.shape(unaries)[0],
            tf.shape(unaries)[1],
            tf.shape(unaries)[2],
            tf.shape(unaries)[3]
        )

        #Spatial filtering
        pos_feature = self.pos_feature(bs)
        spatial_filter = self._create_convolutional_filters(pos_feature)

        #Bilateral filtering((pi,pi),(Ii,Ij)),self.filter_size
        color_pos_feature = self.color_pos_feature(rgb)
        bilateral_filter = self._create_convolutional_filters(color_pos_feature)

        all_ones = tf.ones_like(unaries)

        #norm
        spatial_norm=self.compute_gaussian(all_ones, spatial_filter)
        bilateral_norm=self.compute_gaussian(all_ones, bilateral_filter)

        q_values = unaries

        for i in range(self.num_iterations):
            softmax_out = tf.nn.softmax(q_values, 0)
            # Spatial filtering
            spatial_out = self.compute_gaussian(softmax_out, spatial_filter)
            if self.normalize:
                spatial_out = spatial_out / (spatial_norm+1e-20)

            # Bilateral filtering
            bilateral_out = self.compute_gaussian(softmax_out, bilateral_filter)
            if self.normalize:
                bilateral_out = bilateral_out / (bilateral_norm+1e-20)

            # Weighting filter outputs

            spatial_ker_weights=tf.tile(tf.expand_dims(self.spatial_ker_weights, 0), (bs, 1, 1))
            bilateral_ker_weights=tf.tile(tf.expand_dims(self.bilateral_ker_weights, 0), (bs, 1, 1))

            message_passing = (tf.matmul(spatial_ker_weights,
                                         tf.reshape(spatial_out, (bs, c, -1))) +
                               tf.matmul(bilateral_ker_weights,
                                         tf.reshape(bilateral_out, (bs, c, -1))))

            # Compatibility transform
            compatibility_matrix=tf.tile(tf.expand_dims(self.compatibility_matrix, 0), (bs, 1, 1))
            pairwise = tf.matmul(compatibility_matrix, message_passing)

            # Adding unary potentials,h,w
            if self.blur>1:
                h_b=tf.shape(spatial_out)[2]
                w_b=tf.shape(spatial_out)[3]
                pairwise = tf.reshape(pairwise, (bs, c, h_b, w_b))
                pairwise = tf.transpose(pairwise,perm=(0,2,3,1))
                pairwise = tf.image.resize(pairwise,(h,w),method='bilinear')
                pairwise = tf.transpose(pairwise,perm=(0,3,1,2))
            else:
                pairwise = tf.reshape(pairwise, (bs, c, h, w))

            q_values = unaries - pairwise
        return tf.transpose(q_values, (0, 2, 3, 1)) # (b, h, w, c)
    
if __name__ == "__main__":
    # tf.debugging.set_log_device_placement(True)

    img = tf.random.uniform((4,512,512,3))
    logits = tf.random.normal((4,512,512,2))
    
    layer = ConvCRF((512,512), 7, 1, 2, 160., 3., 3., 10)
    
    out = layer((logits, img))
    print(f"{out.shape=}")

    target = tf.ones_like(out)
    with tf.GradientTape() as tape:
        tape.watch(logits)
        out = layer((logits, img))
        loss = tf.reduce_mean(tf.abs(target - out))

    grads = tape.gradient(loss, layer.trainable_variables)
    print(f"{grads[0].shape=}")