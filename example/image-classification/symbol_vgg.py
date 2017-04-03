"""References:

Simonyan, Karen, and Andrew Zisserman. "Very deep convolutional networks for
large-scale image recognition." arXiv preprint arXiv:1409.1556 (2014).
"""
import find_mxnet
import mxnet as mx
def get_conv(data, kernel, pad, num_filter, name, bn_momentum=0.9, stride=(1,1), with_act=True, with_bn=True, act_type='relu', w_project_method=0, q_method=0):
    conv = mx.symbol.Convolution(
        name=name,
        data=data,
        num_filter=num_filter,
        kernel=kernel,
        stride=stride,
        pad=pad,
        no_bias=False,   
        w_project_method=w_project_method
    )
    bn = mx.symbol.BatchNorm(
        name=name + '_bn',
        data=conv
    )if with_bn else conv
    return (
        # It's better to remove ReLU here
        # https://github.com/gcr/torch-residual-networks
        
        mx.symbol.Activation(name=name + '_act', data=bn, act_type=act_type, q_method=q_method)
        if with_act else bn
    )
def get_symbol(num_classes = 1000):
    ## define alexnet
    data = mx.symbol.Variable(name="data")
    # group 1
    conv1_1 = get_conv(data=data, kernel=(3, 3), pad=(1, 1),stride=(2,2), num_filter=64, with_bn=True, name="conv1_1")
    #pool1 = mx.sym.Pooling(data=conv1_1, kernel=(2, 2), stride=(2, 2), pool_type="max", name="pool1")
    conv1_2 = get_conv(data=conv1_1, kernel=(3, 3), pad=(1, 1), stride=(2,2), num_filter=64, with_bn=True, name="conv1_2")
    # group 2
    conv2_1 = get_conv(
        data=conv1_2, kernel=(3, 3), pad=(1, 1), num_filter=128,w_project_method=1, q_method=0, act_type="sign", name="conv2_1")
    conv2_2 = get_conv(
        data=conv2_1, kernel=(3, 3), pad=(1, 1), stride=(2,2), num_filter=128,w_project_method=1, q_method=0, act_type="sign", name="conv2_2")
    #pool2 = mx.symbol.Pooling(data=relu2_1, pool_type="max", kernel=(2, 2), stride=(2,2), name="pool2")
    # group 3
    conv3_1 = get_conv(
        data=conv2_2, kernel=(3, 3), pad=(1, 1), num_filter=256, w_project_method=1, q_method=0, act_type="sign", name="conv3_1")
    conv3_2 = get_conv(
        data=conv3_1, kernel=(3, 3), pad=(1, 1), num_filter=256, w_project_method=1, q_method=0, act_type="sign", name="conv3_2")
    conv3_3 = get_conv(
        data=conv3_2, kernel=(3, 3), pad=(1, 1), stride=(2,2), num_filter=256, w_project_method=1,  q_method=0, act_type="sign", name="conv3_3")
    #pool3 = mx.symbol.Pooling(data=relu3_2, pool_type="max", kernel=(2, 2), stride=(2,2), name="pool3")
    # group 4
    conv4_1 = get_conv(
        data=conv3_3, kernel=(3, 3), pad=(1, 1), num_filter=512, w_project_method=1, q_method=0, act_type="sign", name="conv4_1")
    conv4_2 = get_conv(
        data=conv4_1, kernel=(3, 3), pad=(1, 1), num_filter=512, w_project_method=1, q_method=0, act_type="sign", name="conv4_2")
    conv4_3 = get_conv(
        data=conv4_2, kernel=(3, 3), pad=(1, 1), stride=(2,2), num_filter=512, w_project_method=1,  q_method=0, act_type="sign", name="conv4_3")
    #pool4 = mx.symbol.Pooling(data=relu4_2, pool_type="max", kernel=(2, 2), stride=(2,2), name="pool4")
    # group 5
    conv5_1 = get_conv(
        data=conv4_3, kernel=(3, 3), pad=(1, 1), num_filter=512, w_project_method=1, q_method=0, act_type="sign", name="conv5_1")
    conv5_2 = get_conv(
        data=conv5_1, kernel=(3, 3), pad=(0, 0), num_filter=512, w_project_method=1, q_method=0, act_type="sign", name="conv5_2")
    conv5_3 = get_conv(
        data=conv5_2, kernel=(3, 3), pad=(0, 0), num_filter=512, w_project_method=1, q_method=0, act_type="sign", name="conv5_3")
    #pool5 = mx.symbol.Pooling(data=relu5_2, pool_type="max", kernel=(2, 2), stride=(2,2), name="pool5")
    # group 6
    flatten = mx.symbol.Flatten(data=conv5_3, name="flatten")
    fc6 = mx.symbol.FullyConnected(data=flatten, num_hidden=3072, w_project_method=1, name="fc6")
    bn6 = mx.symbol.BatchNorm(name='fc6_bn',data=fc6)
    relu6 = mx.symbol.Activation(data=bn6, act_type="sign", q_method=0, name="relu6")
    # group 7
    #fc7 = mx.symbol.FullyConnected(data=relu6*0.001, num_hidden=3072, w_project_method=1, name="fc7")
    #bn7 = mx.symbol.BatchNorm(name='fc7_bn',data=fc7)
    #relu7 = mx.symbol.Activation(data=bn7, act_type="sign", q_method=0, name="relu7")
    # output
    fc8 = mx.symbol.FullyConnected(data=relu6, num_hidden=num_classes, name="fc8")
    bn8 = mx.symbol.BatchNorm(name='fc8_bn',data=fc8)
    softmax = mx.symbol.SoftmaxOutput(data=bn8, name='softmax')
    return softmax
