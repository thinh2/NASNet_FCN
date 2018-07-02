import tensorflow as tf
import os
import numpy as np
from utils import *
import cv2
from glob import glob
from process_data import DataProvider,gen_batches, gen_random_batches
MAX_ITER = int(1e5)

class Model(object):
    def __init__(self, ckpt_dir, log_dir, batch_size=2, input_height=224, input_width=224, num_channels=3,
                    num_classes=6, num_epoch=10, learning_rate=0.00001):

        self.batch_size = batch_size
        self.input_height = input_height
        self.input_width = input_width
        self.num_channels = num_channels
        self.num_classes  = num_classes
        self.sess  = tf.Session()

        self.log_dir = log_dir
        self.num_epoch = num_epoch
        self.learning_rate = learning_rate
        self.ckpt_dir = ckpt_dir

    def build_model(self, model_version):
        self.input_image = tf.placeholder(tf.float32, shape=[self.batch_size, self.input_height, 
                                        self.input_width, self.num_channels], name='input')

        self.ground_truth = tf.placeholder(tf.float32, shape=[self.batch_size, self.input_height,
                                        self.input_width, 1] , name='ground_truth')

        self.model = model_version(self.input_image, self.batch_size, self.ground_truth, self.learning_rate)

        #Rewrite Loss 
 	
        #gts = tf.reshape(self.ground_truth, shape=[self.batch_size, self.input_height, self.input_width])
        #gts = tf.cast(gts, dtype=tf.int32)
        #self.model['loss'] = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        #                                    labels=gts, logits=self.model['upscore']))

        #self.model['optimizer'] = tf.train.AdamOptimizer(self.learning_rate, beta1=0.05).minimize(self.model['loss'])
        #self.model['var_list']  = tf.trainable_variables()
        #self.model['summary_op'] = tf.summary.scalar('loss', self.model['loss'])

    def train(self):
	
        init_op = tf.global_variables_initializer()
        self.saver = tf.train.Saver()
        #self.saver.restore(self.sess, './checkpoint/fcn_8/model-100000')
        self.sess.run(init_op)
        print ("Loading success")
        print ("Start training")
        writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)
        train_X, train_Y, valid_X, valid_Y = DataProvider("./ISPRS_semantic_labeling_Vaihingen").load_data(images_from_each=500, ground_truth=True)

        counter = 0
        for _ in range(self.num_epoch):
            for x,y in gen_batches(train_X, train_Y, batch_size=self.batch_size):              
                #x, y = train_data.get_next_batch(batch_size=self.batch_size)
                _, loss, summary_str = self.sess.run([self.model['optimizer'], self.model['loss'], self.model['summary_op']], 
                                                            feed_dict={self.input_image : x, self.ground_truth : y})
                counter += 1      
                if counter % 10 == 0:
                    writer.add_summary(summary_str, counter)

                if counter % 5000 == 0 :
                    #valid_test
                    #print "valid mode"
                    self.save_model(counter)
                    #print "save done"
                    score = self.valid_test(valid_X, valid_Y)
                    with open('{0}/fcn8s_valid_and_loss.txt'.format(str(self.log_dir)), "a") as myfile:
                        myfile.write("Overall Accuraccy : {0:.5f}\n".format(score))
                        myfile.write("Iter {0} : Loss {1}\n".format(str(counter), str(loss)))

                    x, y = gen_random_batches(valid_X, valid_Y, batch_size=self.batch_size)
                    self.visualize(x, y, counter)


    def save_model(self, counter):
        model_name = 'model'
        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)
        self.saver.save(self.sess, os.path.join(self.ckpt_dir, model_name), global_step=counter)

    def valid_test(self, valid_X, valid_Y):
        preds = []
        gts = []
        for x,y in gen_batches(valid_X, valid_Y, batch_size=self.batch_size, shuffle=False):
            #x, y = valid_data.get_next_batch()
            _pred = self.sess.run(self.model['predict'], feed_dict={self.input_image:x, self.ground_truth:y })
            gts.append(y)
            preds.append(_pred)
        
        score = get_score(gts, preds)
        return score

    def visualize(self, x, y, counter):

        pred = self.sess.run(self.model['predict'], feed_dict={self.input_image:x})
        #print pred.shape
        for id in xrange(pred.shape[0]):
            image_save(labels_2_rgb(pred[id]) , "{2}/{0}_{1}_predict.tiff".format(str(counter), str(id), str(self.log_dir)))
            image_save(x[id] , "{2}/{0}_{1}_top.tiff".format(str(counter), str(id), str(self.log_dir)))
            image_save(labels_2_rgb(y[id]) , "{2}/{0}_{1}_ground_truth.tiff".format(str(counter), str(id), str(self.log_dir)))
            
    def inference(self, fcn_version, ckpt_path, overlap_size = 112):
        #Refactor inference
        self.build_model(fcn_version)
        saver = tf.train.Saver()
        saver.restore(self.sess, ckpt_path)

        print "Start loading data"

        data_provider = DataProvider('./ISPRS_semantic_labeling_Vaihingen')
        for idx in DataProvider.test_idx:
            test_data,test_data_info = data_provider.get_chunk_data(idx, overlap_size=overlap_size)
            print "Load test data success"

            print test_data.shape
            preds = []
            for i in xrange(test_data.shape[0]):
                tmp = self.sess.run(self.model['upscore'], feed_dict={self.input_image : test_data[i : i + self.batch_size]})
                preds.append(tmp[0])
            
            data_provider.merge_chunks(idx, np.array(preds), test_data_info)


#if __name__ == '__main__':
#    fcn = FCN('./checkpoint/fcn_32', './logs/fcn_32', batch_size=1)
#    fcn.inference(fcn32s, './checkpoint/fcn_32/model-128000', overlap_size=50)
