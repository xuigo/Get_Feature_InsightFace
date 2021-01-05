import os
import math
import cv2
import numpy as np
import tensorflow as tf
from scipy import misc
from model import get_embd


IMAGE_SIZE=112
PATH1='./images/1.jpg'
PATH2='./images/2.jpg'
MODEL_PATH='checkpoints/best-m-1006000'

class get_embd_insightFace:

    """ Tensorflow implementation of getting face feature using InsightFace"""
    
    def __init__(self,src1,src2):
        self.src1=src1
        self.src2=src2
        self.image_size=IMAGE_SIZE
        self.model_path=MODEL_PATH
        self.load_image()
        self.build_graph()
        self.restore()

    def load_image(self):
        self.ims=[]
        for file in [self.src1,self.src2]:
            if not os.path.isfile(file):
                raise ValueError('[!] %s doesn\'t exist!'%file)
            img=cv2.imread(file)  
            img=cv2.resize(img,(256,256))  # Note: This is used in my project,    
            img = img[35:223, 32:220,:]    # if doesn’t suit you, just comment it out.
            img=cv2.resize(img,(self.image_size,self.image_size)) 
            img = img/127.5-1.0
            self.ims.append(img)
        return np.array(self.ims)

    def build_graph(self):
        self.images = tf.placeholder(dtype=tf.float32, shape=[None, self.image_size, self.image_size, 3], name='input_image')
        self.embds, _ = get_embd(self.images, is_training_dropout=False, is_training_bn=False)
        _tf_config = tf.ConfigProto(allow_soft_placement=True)
        _tf_config.gpu_options.allow_growth = True
        self.sess=tf.Session(config=_tf_config)

    def restore(self):
        saver = tf.train.Saver(var_list=tf.trainable_variables())
        saver = tf.train.Saver()
        saver.restore(self.sess, self.model_path)

    def calc_distance(self,features,distance_metric=1):
        # here using arcosine as the cost-function
        embeddings1,embeddings2=features[0][np.newaxis,:],features[1][np.newaxis,:]
        if distance_metric==0:
            # Euclidian distance
            embeddings1 = embeddings1/np.linalg.norm(embeddings1, axis=1, keepdims=True)
            embeddings2 = embeddings2/np.linalg.norm(embeddings2, axis=1, keepdims=True)
            diff = np.subtract(embeddings1, embeddings2)
            dist = np.sum(np.square(diff),1)
        elif distance_metric==1:
            # Distance based on cosine similarity
            dot = np.sum(np.multiply(embeddings1, embeddings2), axis=1)
            norm = np.linalg.norm(embeddings1, axis=1) * np.linalg.norm(embeddings2, axis=1)
            similarity = dot/norm
            eps = 1e-6
            if 1.0 < similarity < 1.0 + eps:
                similarity = 1.0
            elif -1.0 - eps < similarity < -1.0:
                similarity = -1.0
            dist = np.arccos(similarity) / math.pi
        else:
            raise 'Undefined distance metric %d' % distance_metric 
        return similarity,dist

    def run(self):
        features=self.sess.run(self.embds,feed_dict={self.images:self.load_image()})
        similarity,dist=self.calc_distance(features)   #similarity: Face similarity (the bigger the more similar); dist: Face cosine angle (the smaller the value, the smaller the angle)
        return similarity,dist

if __name__ == '__main__':
    
    run_instance=get_embd_insightFace(PATH1,PATH2)
    similarity,angle=run_instance.run() # 1-similarity can used as the id-loss
    print('\nFace Similarity is:{}  Face angle is:{}'.format(similarity[0],angle[0]))  
        