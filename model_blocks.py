import tensorflow as tf

_LEAKY_RELU = 0.1
_BATCH_NORM_DECAY = 0.9
_BATCH_NORM_EPSILON = 1e-05

def batch_normalization(inputs, training, use_channels_first):
    """Performs batch normalization using standard parameters.
    
    Args:
        inputs: Input tensor.
        training: A boolean indicating whether the model is in training mode.
        use_channels_first: A boolean indicating whether 'channels_first' format is used.
        
    Returns:
        Output tensor after batch normalization.
    """
    axis = 1 if use_channels_first else 3
    bn = tf.keras.layers.BatchNormalization(
        axis=axis,
        momentum=_BATCH_NORM_DECAY,
        epsilon=_BATCH_NORM_EPSILON,
        scale=True,
        trainable=training)
    
    return bn(inputs, training=training)


def fixed_padding(inputs, kernel_size, data_format):
    """ResNet implementation of fixed padding.
    
    Args:
        inputs: Tensor input to be padded.
        kernel_size: The kernel to be used in the conv2d or max_pool2d.
        data_format: The input format.
        
    Returns:
        A tensor with the same format as the input.
    """
    PADDING_KERNEL_SIZE = kernel_size
    PAD_TOTAL = PADDING_KERNEL_SIZE - 1
    PAD_BEG = PAD_TOTAL // 2
    PAD_END = PAD_TOTAL - PAD_BEG
    
    if data_format == 'channels_first':
        padded_inputs = tf.pad(inputs, [[0, 0], [0, 0],
                                        [PAD_BEG, PAD_END],
                                        [PAD_BEG, PAD_END]])
    elif data_format == 'channels_last':
        padded_inputs = tf.pad(inputs, [[0, 0], [PAD_BEG, PAD_END],
                                        [PAD_BEG, PAD_END], [0, 0]])
    else:
        raise ValueError("Invalid data_format. Supported values are 'channels_first' and 'channels_last'.")
    
    return padded_inputs


def conv2d_fixed_padding(inputs, filters, kernel_size, data_format, strides=1):
    """Strided 2-D convolution with explicit padding.
    
    Args:
        inputs: Input tensor.
        filters: Number of output filters.
        kernel_size: Size of the convolutional kernel.
        data_format: 'channels_last' or 'channels_first'.
        strides: Stride size for the convolution. Default is 1.
        
    Returns:
        Output tensor after convolution.
    """
    if strides > 1:
        inputs = fixed_padding(inputs, kernel_size, data_format)
    
    conv_layer = tf.keras.layers.Conv2D(
        filters=filters, kernel_size=kernel_size,
        strides=strides, padding=('SAME' if strides == 1 else 'VALID'),
        use_bias=False, data_format=data_format)

    return conv_layer(inputs)


def darknet53_residual_block(inputs, filters, training, data_format, strides=1):
    """Creates a residual block for Darknet.
    
    Args:
        inputs: Input tensor.
        filters: Number of filters in the convolutional layers.
        training: A boolean indicating whether the model is in training mode.
        data_format: 'channels_first' or 'channels_last'.
        strides: Stride size for the convolutional layers. Default is 1.
        
    Returns:
        Output tensor after the residual block.
    """
    shortcut = inputs
    use_channels_first = data_format == 'channels_first'

    inputs = conv2d_fixed_padding(
        inputs, filters=filters, kernel_size=1, strides=strides,
        data_format=data_format)
    inputs = batch_normalization(inputs, training=training, use_channels_first=use_channels_first)
    inputs = tf.nn.leaky_relu(inputs, alpha=_LEAKY_RELU)

    inputs = conv2d_fixed_padding(
        inputs, filters=2 * filters, kernel_size=3, strides=strides,
        data_format=data_format)
    inputs = batch_normalization(inputs, training=training, use_channels_first=use_channels_first)
    inputs = tf.nn.leaky_relu(inputs, alpha=_LEAKY_RELU)

    inputs += shortcut

    return inputs


def darknet53(inputs, training, data_format):
    """Creates Darknet53 model for feature extraction.
    
    Args:
        inputs: Input tensor.
        training: A boolean indicating whether the model is in training mode.
        data_format: 'channels_first' or 'channels_last'.
        
    Returns:
        Three tensors: route1, route2, and the final feature tensor.
    """
    use_channels_first = data_format == 'channels_first'
    
    filters = [32, 64, 128, 256, 512, 1024]

    inputs = conv2d_fixed_padding(inputs, filters=filters[0], kernel_size=3,
                                  data_format=data_format)
    inputs = batch_normalization(inputs, training=training, use_channels_first=use_channels_first)
    inputs = tf.nn.leaky_relu(inputs, alpha=_LEAKY_RELU)
    inputs = conv2d_fixed_padding(inputs, filters=filters[1], kernel_size=3,
                                  strides=2, data_format=data_format)
    inputs = batch_normalization(inputs, training=training, use_channels_first=use_channels_first)
    inputs = tf.nn.leaky_relu(inputs, alpha=_LEAKY_RELU)

    inputs = darknet53_residual_block(inputs, filters=filters[0], training=training,
                                      data_format=data_format)

    inputs = conv2d_fixed_padding(inputs, filters=filters[2], kernel_size=3,
                                  strides=2, data_format=data_format)
    inputs = batch_normalization(inputs, training=training, use_channels_first=use_channels_first)
    inputs = tf.nn.leaky_relu(inputs, alpha=_LEAKY_RELU)

    for _ in range(2):
        inputs = darknet53_residual_block(inputs, filters=filters[1],
                                          training=training,
                                          data_format=data_format)

    inputs = conv2d_fixed_padding(inputs, filters=filters[3], kernel_size=3,
                                  strides=2, data_format=data_format)
    inputs = batch_normalization(inputs, training=training, use_channels_first=use_channels_first)
    inputs = tf.nn.leaky_relu(inputs, alpha=_LEAKY_RELU)

    for _ in range(8):
        inputs = darknet53_residual_block(inputs, filters=filters[2],
                                          training=training,
                                          data_format=data_format)

    route1 = inputs

    inputs = conv2d_fixed_padding(inputs, filters=filters[4], kernel_size=3,
                                  strides=2, data_format=data_format)
    inputs = batch_normalization(inputs, training=training, use_channels_first=use_channels_first)
    inputs = tf.nn.leaky_relu(inputs, alpha=_LEAKY_RELU)

    for _ in range(8):
        inputs = darknet53_residual_block(inputs, filters=filters[3],
                                          training=training,
                                          data_format=data_format)

    route2 = inputs

    inputs = conv2d_fixed_padding(inputs, filters=filters[5], kernel_size=3,
                                  strides=2, data_format=data_format)
    inputs = batch_normalization(inputs, training=training, use_channels_first=use_channels_first)
    inputs = tf.nn.leaky_relu(inputs, alpha=_LEAKY_RELU)

    for _ in range(4):
        inputs = darknet53_residual_block(inputs, filters=filters[4],
                                          training=training,
                                          data_format=data_format)

    return route1, route2, inputs


def yolo_convolution_block(inputs, filters, training, data_format):
    """Creates convolution operations layer used after Darknet.
    
    Args:
        inputs: Input tensor.
        filters: Number of filters for the convolutional layers.
        training: A boolean indicating whether the model is in training mode.
        data_format: 'channels_first' or 'channels_last'.
        
    Returns:
        Two tensors: route and the final feature tensor.
    """
    use_channels_first = data_format == 'channels_first'
    
    def conv_bn_leaky(inputs, filters, kernel_size):
        inputs = conv2d_fixed_padding(inputs, filters=filters, kernel_size=kernel_size,
                                      data_format=data_format)
        inputs = batch_normalization(inputs, training=training, use_channels_first=use_channels_first)
        inputs = tf.nn.leaky_relu(inputs, alpha=_LEAKY_RELU)
        return inputs
    
    inputs = conv_bn_leaky(inputs, filters, kernel_size=1)
    inputs = conv_bn_leaky(inputs, 2 * filters, kernel_size=3)
    inputs = conv_bn_leaky(inputs, filters, kernel_size=1)
    inputs = conv_bn_leaky(inputs, 2 * filters, kernel_size=3)
    inputs = conv_bn_leaky(inputs, filters, kernel_size=1)

    route = inputs
    
    inputs = conv_bn_leaky(inputs, 2 * filters, kernel_size=3)
    
    return route, inputs


def yolo_layer(inputs, n_classes, anchors, img_size, data_format):
    """Creates YOLO final detection layer.
    
    Args:
        inputs: Tensor input.
        n_classes: Number of labels.
        anchors: A list of anchor sizes.
        img_size: The input size of the model.
        data_format: The input format.
        
    Returns:
        Tensor output.
    """
    n_anchors = len(anchors)
    c2d = tf.keras.layers.Conv2D(
        filters=n_anchors * (5 + n_classes), kernel_size=1, strides=1,
        use_bias=True, data_format=data_format)

    inputs = c2d(inputs)

    shape = inputs.get_shape().as_list()
    grid_shape = shape[2:4] if data_format == 'channels_first' else shape[1:3]
    if data_format == 'channels_first':
        inputs = tf.transpose(inputs, [0, 2, 3, 1])
    inputs = tf.reshape(inputs, [-1, n_anchors * grid_shape[0] * grid_shape[1], 5 + n_classes])

    strides = (img_size[0] // grid_shape[0], img_size[1] // grid_shape[1])

    box_centers, box_shapes, confidence, classes = \
        tf.split(inputs, [2, 2, 1, n_classes], axis=-1)

    x = tf.range(grid_shape[0], dtype=tf.float32)
    y = tf.range(grid_shape[1], dtype=tf.float32)
    x_offset, y_offset = tf.meshgrid(x, y)
    x_offset = tf.reshape(x_offset, (-1, 1))
    y_offset = tf.reshape(y_offset, (-1, 1))
    x_y_offset = tf.concat([x_offset, y_offset], axis=-1)
    x_y_offset = tf.tile(x_y_offset, [1, n_anchors])
    x_y_offset = tf.reshape(x_y_offset, [1, -1, 2])
    box_centers = tf.nn.sigmoid(box_centers)
    box_centers = (box_centers + x_y_offset) * strides

    anchors = tf.tile(anchors, [grid_shape[0] * grid_shape[1], 1])
    box_shapes = tf.exp(box_shapes) * tf.cast(anchors, dtype=tf.float32)

    confidence = tf.nn.sigmoid(confidence)
    classes = tf.nn.sigmoid(classes)

    inputs = tf.concat([box_centers, box_shapes, confidence, classes], axis=-1)

    return inputs


def upsample(inputs, out_shape, data_format):
    """Upsamples to `out_shape` using nearest neighbor interpolation.
    
    Args:
        inputs: Input tensor.
        out_shape: Desired output shape (height, width).
        data_format: 'channels_first' or 'channels_last'.
        
    Returns:
        Upsampled tensor.
    """
    if data_format == 'channels_first':
        new_height, new_width = out_shape[3], out_shape[2]
    else:
        new_height, new_width = out_shape[2], out_shape[1]

    inputs = tf.image.resize(inputs, (new_height, new_width), method='nearest')

    if data_format == 'channels_first':
        inputs = tf.transpose(inputs, [0, 3, 1, 2])

    return inputs


def build_boxes(inputs):
    """Computes top left and bottom right points of the boxes.
    
    Args:
        inputs: Tensor containing center_x, center_y, width, height, confidence, and classes.
        
    Returns:
        Tensor of computed bounding boxes.
    """
    center_x, center_y, width, height, confidence, classes = \
        tf.split(inputs, [1, 1, 1, 1, 1, -1], axis=-1)

    top_left_x = center_x - width / 2
    top_left_y = center_y - height / 2
    bottom_right_x = center_x + width / 2
    bottom_right_y = center_y + height / 2

    boxes = tf.concat([top_left_x, top_left_y,
                       bottom_right_x, bottom_right_y,
                       confidence, classes], axis=-1)

    return boxes


def non_max_suppression(inputs, n_classes, max_output_size, iou_threshold,
                        confidence_threshold):
    """Performs non-max suppression separately for each class.
    
    Args:
        inputs: Tensor input.
        n_classes: Number of classes.
        max_output_size: Max number of boxes to be selected for each class.
        iou_threshold: Threshold for the IOU.
        confidence_threshold: Threshold for the confidence score.
        
    Returns:
        A list containing class-to-boxes dictionaries for each sample in the batch.
    """
    batch = tf.unstack(inputs)
    boxes_dicts = []
    for boxes in batch:
        boxes = tf.boolean_mask(boxes, boxes[:, 4] > confidence_threshold)
        classes = tf.argmax(boxes[:, 5:], axis=-1)
        classes = tf.expand_dims(tf.cast(classes, dtype=tf.float32), axis=-1)
        boxes = tf.concat([boxes[:, :5], classes], axis=-1)

        boxes_dict = dict()
        for cls in range(n_classes):
            mask = tf.equal(boxes[:, 5], cls)
            mask_shape = mask.get_shape()
            if mask_shape.ndims != 0:
                class_boxes = tf.boolean_mask(boxes, mask)
                boxes_coords, boxes_conf_scores, _ = tf.split(class_boxes,
                                                              [4, 1, -1],
                                                              axis=-1)
                boxes_conf_scores = tf.reshape(boxes_conf_scores, [-1])
                indices = tf.image.non_max_suppression(boxes_coords,
                                                       boxes_conf_scores,
                                                       max_output_size,
                                                       iou_threshold)
                class_boxes = tf.gather(class_boxes, indices)
                boxes_dict[cls] = class_boxes[:, :5]

        boxes_dicts.append(boxes_dict)

    return boxes_dicts

