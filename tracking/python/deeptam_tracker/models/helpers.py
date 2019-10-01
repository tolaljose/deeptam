import lmbspecialops as sops
import tensorflow as tf
import math

""" 
helper.py
---------

functions:
    1. convert_NCHW_to_NHWC()
    2. convert_NHWC_to_NCHW()
    3. myLeakyRelu()
    4. default_weights_initializer()
    5. conv2d()
    6. convrelu()
    7. convrelu2()
    8. fcrelu()
    9. scale_tensor()
    10. resize_nearest_neighbor_NCHW()
    11. resize_area_NCHW()
    12. apply_motion_increment()
"""


def convert_NCHW_to_NHWC(inp):
    """
    1. convert_NCHW_to_NHWC():
        - Note:
            - N: number of images in the batch
            - H: height of the image
            - W: width of the image
            - C: number of channels of the image (ex: 3 for RGB, 1 for grayscale...)
            - NVIDIA recommends NHCW (https://docs.nvidia.com/deeplearning/sdk/
                                      pdf/Deep-Learning-Performance-Guide.pdf)

        - input: tensor (in NCHW)
        - return: tensor (in NHWC)
        - fuctionality: convert the input tensor of the format NCHW
             to NHCW.
             N C H W  =====>  N H W C
            [0 1 2 3]        [0 2 3 1]
    """
    return tf.transpose(inp, [0, 2, 3, 1])


def convert_NHWC_to_NCHW(inp):
    """
    2. convert_NHWC_to_NCHW():
        - Note:
            - N: number of images in the batch
            - H: height of the image
            - W: width of the image
            - C: number of channels of the image (ex: 3 for RGB, 1 for grayscale...)
            - NVIDIA recommends NHCW
            (Ref: https://docs.nvidia.com/deeplearning/sdk/pdf/
             Deep-Learning-Performance-Guide.pdf)

        - input: tensor (in NHWC)
        - return: tensor (in NCHW)
        - fuctionality: convert the input tensor of the format NHWC
             to NCHW.

             N H W C  =====>  N C H W
            [0 1 2 3]        [0 3 1 2]
    """
    return tf.transpose(inp, [0, 3, 1, 2])


def myLeakyRelu(x):
    """
    3. myLeakyRelu()
        - Note:
            Leaky ReLU. Leaky ReLUs are one attempt to fix the “dying ReLU” problem.
            Instead of the function being zero when x < 0, a leaky ReLU will instead
            have a small negative slope (of 0.01, or so). That is, the function
            computes:
                f(x) = 1(x<0)(αx) + 1(x>=0)(x)
                where α is a small constant.
            (Ref: http://cs231n.github.io/neural-networks-1/
                  https://arxiv.org/pdf/1811.03378.pdf)

        - input: Tensor -representing preactivation values. Must be one of the
            following types: `float16`, `float32`, `float64`, `int32`, `int64`.
        - return: activation value
        - functionality: Leaky ReLU with leak factor 0.1
    """
    # return tf.maximum(0.1*x,x)
    # leaky relu from lmbspecialops deeptam branch
    return sops.leaky_relu(x, leak=0.1)
    # return tf.nn.leaky_relu(x, alpha=0.1)


def default_weights_initializer():
    """
    4. default_weights_initializer()
        - Note:
            Samples are drawn from a truncated normal distribution
            centered on zero,
            with stddev = sqrt(scale / n),
            where n is the number of input units in the weight tensor
            (Ref: http://tensorflow.biotecan.com/python/Python_1.8/tensorflow.google.cn
                 /api_docs/python/tf/variance_scaling_initializer.html)

        - input: None
        - return: variance_scaling_initializer Object with scale = 2.0
        - functionality:
    """
    return tf.variance_scaling_initializer(scale=2.0)


# Lal : included the name
# to consider using tensorboard
def conv2d(inputs, num_outputs, kernel_size, data_format, padding=None, **kwargs):
    """
    5. conv2d()
        - Note:
            2D convolution layer (e.g. spatial convolution over images).
            This layer creates a convolution kernel that is convolved (actually
            cross-correlated) with the layer input to produce a tensor of outputs.
            (Ref: http://tensorflow.biotecan.com/python/Python_1.8/
                     tensorflow.google.cn/api_docs/python/tf/layers/Conv2D.html)
        - input:
            - inputs: input tensor(s)
            - num_outputs:
                Integer, the dimensionality of the output space
                (i.e. the number of filters in the convolution)
            - kernel_size:
                 An integer or tuple/list of 2 integers,
                 specifying the height and width of the 2D convolution window.
                 Can be a single integer to specify the same value
                 for all spatial dimensions.
            - data_format:
                A string, one of channels_last (default) or channels_first.
                The ordering of the dimensions in the inputs.
                channels_last corresponds to inputs with shape
                    (batch, height, width, channels)
                while channels_first corresponds to inputs with shape
                    (batch, channels, height, width).
            - padding: One of "valid" or "same" (case-insensitive).
                default is "same"
        - return: Output tensor(s)
        - functionality: Convolution with 'same' padding if 'valid' not
            mentioned explicitly
    """
    # padding "same" unless not explitly mentioned
    if padding is None:
        padding = 'same'
    # with kernel initialized considering the samples are drawn
    # from a truncated normal distribution centered on zero
    """
    soon to depreciated - tf.keras.layers/ tf.nn.layers to be used
    """
    return tf.layers.conv2d(
        inputs=inputs,
        filters=num_outputs,
        kernel_size=kernel_size,
        kernel_initializer=default_weights_initializer(),
        padding=padding,
        data_format=data_format,
        **kwargs,
    )


# Lal : included the name
# to consider using tensorboard
def convrelu(inputs, num_outputs, kernel_size, data_format, activation=None, **kwargs):
    """
    6. convrelu()
        - Note:
            2D convolution layer (e.g. spatial convolution over images).
            This layer creates a convolution kernel that is convolved (actually
            cross-correlated) with the layer input to produce a tensor of outputs.
            (Ref: http://tensorflow.biotecan.com/python/Python_1.8/
                     tensorflow.google.cn/api_docs/python/tf/layers/Conv2D.html)
        - input:
            - inputs: input tensor(s)
            - num_outputs:
                Integer, the dimensionality of the output space
                (i.e. the number of filters in the convolution)
            - kernel_size:
                 An integer or tuple/list of 2 integers,
                 specifying the height and width of the 2D convolution window.
                 Can be a single integer to specify the same value
                 for all spatial dimensions.
            - data_format:
                A string, one of channels_last (default) or channels_first.
                The ordering of the dimensions in the inputs.
                channels_last corresponds to inputs with shape
                    (batch, height, width, channels)
                while channels_first corresponds to inputs with shape
                    (batch, channels, height, width).
            - activation:
                Activation function. Set it to None to maintain a LeakyRelu activation.
        - return: Output tensor(s)
        - functionality: Shortcut for a single convolution layer with LeakyRelu,
            if other activation not specified.
    """
    # activation LeakyRelu unless not explicitly mentioned
    if activation is None:
        activation = myLeakyRelu

    return conv2d(inputs, num_outputs, kernel_size, data_format, activation=activation, **kwargs)


def convrelu2(inputs, num_outputs, kernel_size, name, stride, data_format,
              padding=None, activation=None, **kwargs):
    """
    7. convrelu2()
        - Note:
            2D convolution layer (e.g. spatial convolution over images).
            This layer creates a convolution kernel that is convolved (actually
            cross-correlated) with the layer input to produce a tensor of outputs.
            (Ref: http://tensorflow.biotecan.com/python/Python_1.8/
                     tensorflow.google.cn/api_docs/python/tf/layers/Conv2D.html)
        - input:
            - inputs: input tensor(s)
            - num_outputs:
                Integer, the dimensionality of the output space
                (i.e. the number of filters in the convolution)
            - kernel_size:
                 An integer or tuple/list of 2 integers,
                 specifying the height and width of the 2D convolution window.
                 Can be a single integer to specify the same value
                 for all spatial dimensions.
            - name
            - stride:
                 An integer,
                 specifying the strides of the convolution along the height and width.
            - data_format
            - padding: One of "valid" or "same" (case-insensitive).
                Set it None for "same"
            - activation: Activation function. Set it to None to maintain
                a LeakyRelu activation.
        - return: Output tensor(s)
        - functionality: Shortcut for two convolution+relu with 1D filter kernels
        int or (int,int)
        If num_outputs is a tuple then the first element is the number of
        outputs for the 1d filter in y direction and the second element is
        the final number of outputs.
    """

    # separate x and y direction output sizes for the separated convolutions
    # in each direction
    if isinstance(num_outputs, (tuple, list)):
        num_outputs_y = num_outputs[0]
        num_outputs_x = num_outputs[1]
    else:
        num_outputs_y = num_outputs
        num_outputs_x = num_outputs
    # separate x and y direction kernel sizes for the separated convolutions
    # in each direction
    if isinstance(kernel_size, (tuple, list)):
        kernel_size_y = kernel_size[0]
        kernel_size_x = kernel_size[1]
    else:
        kernel_size_y = kernel_size
        kernel_size_x = kernel_size
    # padding "same" unless not explitly mentioned
    if padding is None:
        padding = 'same'
    # activation LeakyRelu unless not explicitly mentioned
    if activation is None:
        activation = myLeakyRelu
    # the convolution in the y direction
    # with kernel initialized considering the samples are drawn
    # from a truncated normal distribution centered on zero
    tmp_y = tf.layers.conv2d(
        inputs=inputs,
        filters=num_outputs_y,
        kernel_size=[kernel_size_y, 1],
        strides=[stride, 1],
        padding=padding,
        activation=activation,
        kernel_initializer=default_weights_initializer(),
        data_format=data_format,
        name=name + 'y',
        **kwargs,
    )
    # using the y convoluted tensor
    # the convolution in the x direction
    # with kernel initialized considering the samples are drawn
    # from a truncated normal distribution centered on zero
    return tf.layers.conv2d(
        inputs=tmp_y,
        filters=num_outputs_x,
        kernel_size=[1, kernel_size_x],
        strides=[1, stride],
        padding=padding,
        activation=activation,
        kernel_initializer=default_weights_initializer(),
        data_format=data_format,
        name=name + 'x',
        **kwargs,
    )


def fcrelu(inputs, name, num_outputs, weights_regularizer=None, activation=None, **kwargs):
    """
    8. fcrelu()
        - Note: Densely-connected layer.
        - input:
            - inputs: input tensor(s)
            - name
            - num_outputs:  Integer or Long, dimensionality of the output space.
            - weights_regularizer: don't care
            - activation: Activation function. Set it to None to maintain
                a LeakyRelu activation.
        - return: Tensor
        - functionality: creates a fully connected layer
    """
    # activation LeakyRelu unless not explicitly mentioned
    if activation is None:
        activation = myLeakyRelu
    # dense layer with kernel initialized considering the samples are drawn
    # from a truncated normal distribution centered on zero
    return tf.layers.dense(
        inputs=inputs,
        units=num_outputs,
        activation=activation,
        kernel_initializer=default_weights_initializer(),
        name=name,
        **kwargs,
    )


def scale_tensor(inp, scale_factor, data_format='NCHW'):
    """
    9. scale_tensor()
        - Note:
        - input:
            - inp: tensor in BCHW
            - scale_factor:
                signed int (+: upscale, -: downscale)
            - data_format: str
        - return:
        - functionality:
            Down/Up scale the tensor by a factor using nearest neighbor
    """
    if data_format == 'NCHW':
        if scale_factor == 0:
            return inp
        else:
            data_shape = inp.get_shape().as_list()
            # channel_num = data_shape[1]
            height_org = data_shape[2]
            width_org = data_shape[3]
            height_new = int(height_org * math.pow(2, scale_factor))
            width_new = int(width_org * math.pow(2, scale_factor))

            inp_tmp = convert_NCHW_to_NHWC(inp)
            resize_shape_tensor = tf.constant([height_new, width_new], tf.int32)
            inp_resize = tf.image.resize_nearest_neighbor(inp_tmp, resize_shape_tensor)

            return convert_NHWC_to_NCHW(inp_resize)
    else:
        raise Exception('scale_tensor does not support {0} format now.'.format(data_format))


def resize_nearest_neighbor_NCHW(inp, size):
    """
    10. resize_nearest_neighbor_NCHW()
        - Note:
        - input:
            - inp: Tensor
            - size: list with height and width
        - return: Tensor
        - functionality: shortcut for resizing with NCHW format
    """
    if inp.get_shape().as_list()[-2:] == list(size):
        return inp
    else:
        return convert_NHWC_to_NCHW(tf.image.resize_nearest_neighbor(convert_NCHW_to_NHWC(inp),
                                                                     size, align_corners=True))


def resize_area_NCHW(inp, size):
    """
    11. resize_area_NCHW()
        - Note:
        - input:
            - inp: Tensor
            - size: list with height and width
        - return: Tensor
        - functionality: shortcut for resizing with NCHW format
    """
    if inp.get_shape().as_list()[-2:] == list(size):
        return inp
    else:
        return convert_NHWC_to_NCHW(tf.image.resize_area(convert_NCHW_to_NHWC(inp), size, align_corners=True))


def apply_motion_increment(R_prev, t_prev, R_inc, t_inc):
    """
    12. apply_motion_increment()
        - Note:
            - R_next = R_inv*R_prev
            - t_next = t_inc + R_inc*t_prev
        - input:
            - R_prev
            - t_prev
            - t_inc
        - return:
            - R_angleaxis_next
            - t_next
        - functionality: Apply motion increment to previous motion
    """
    R_matrix_prev = sops.angle_axis_to_rotation_matrix(R_prev)
    R_matrix_inc = sops.angle_axis_to_rotation_matrix(R_inc)
    R_matrix_next = tf.matmul(R_matrix_inc, R_matrix_prev)
    R_angleaxis_next = sops.rotation_matrix_to_angle_axis(R_matrix_next)
    t_next = tf.add(t_inc, tf.squeeze(tf.matmul(R_matrix_inc, tf.expand_dims(t_prev, 2)), [2, ]))

    return R_angleaxis_next, t_next


""" End of the script """

