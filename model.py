import tensorflow as tf
import tensorflow.contrib.slim as slim

from backbones import modifiedResNet_v2, ResNet_v2

BACKBONE_TYPE='resnet_v2_m_50'
WEIGHT_DECAY= 5e-4
BN_DECAY= 0.9
KEEP_PROB= 0.4
EMBD_SIZE= 512
OUT_TYPE='E'

def get_embd(inputs, is_training_dropout, is_training_bn,  reuse=False, scope='embd_extractor'):
    with tf.variable_scope(scope, reuse=reuse):
        net = inputs
        end_points = {}
        if BACKBONE_TYPE.startswith('resnet_v2_m'):
            arg_sc = modifiedResNet_v2.resnet_arg_scope(weight_decay=WEIGHT_DECAY, batch_norm_decay=BN_DECAY)
            with slim.arg_scope(arg_sc):
                if BACKBONE_TYPE == 'resnet_v2_m_50':
                    net, end_points = modifiedResNet_v2.resnet_v2_m_50(net, is_training=is_training_bn, return_raw=True)
                elif BACKBONE_TYPE == 'resnet_v2_m_101':
                    net, end_points = modifiedResNet_v2.resnet_v2_m_101(net, is_training=is_training_bn, return_raw=True)
                elif BACKBONE_TYPE == 'resnet_v2_m_152':
                    net, end_points = modifiedResNet_v2.resnet_v2_m_152(net, is_training=is_training_bn, return_raw=True)
                elif BACKBONE_TYPE == 'resnet_v2_m_200':
                    net, end_points = modifiedResNet_v2.resnet_v2_m_200(net, is_training=is_training_bn, return_raw=True)
                else:
                    raise ValueError('Invalid backbone type.')
        elif BACKBONE_TYPE.startswith('resnet_v2'):
            arg_sc = ResNet_v2.resnet_arg_scope(weight_decay=WEIGHT_DECAY, batch_norm_decay=BN_DECAY)
            with slim.arg_scope(arg_sc):
                if BACKBONE_TYPE == 'resnet_v2_50':
                    net, end_points = ResNet_v2.resnet_v2_50(net, is_training=is_training_bn, return_raw=True)
                elif BACKBONE_TYPE == 'resnet_v2_101':
                    net, end_points = ResNet_v2.resnet_v2_101(net, is_training=is_training_bn, return_raw=True)
                elif BACKBONE_TYPE == 'resnet_v2_152':
                    net, end_points = ResNet_v2.resnet_v2_152(net, is_training=is_training_bn, return_raw=True)
                elif BACKBONE_TYPE == 'resnet_v2_200':
                    net, end_points = ResNet_v2.resnet_v2_200(net, is_training=is_training_bn, return_raw=True)
        else:
            raise ValueError('Invalid backbone type.')

        if OUT_TYPE == 'E':
            with slim.arg_scope(arg_sc):
                net = slim.batch_norm(net, activation_fn=None, is_training=is_training_bn)
                net = slim.dropout(net, keep_prob=KEEP_PROB, is_training=is_training_dropout)
                net = slim.flatten(net)
                net = slim.fully_connected(net, EMBD_SIZE, normalizer_fn=None, activation_fn=None)
                net = slim.batch_norm(net, scale=False, activation_fn=None, is_training=is_training_bn)
                end_points['embds'] = net
        else:
            raise ValueError('Invalid out type.')
        
        return net, end_points


        
        