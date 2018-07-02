from keras.applications.nasnet import NASNetLarge, NASNetMobile
from keras.layers import Conv2DTranspose, Concatenate, Add, Conv2D, Input
from keras.optimizers import SGD, Adam
from keras.callbacks import ModelCheckpoint, Callback
from keras.models import Model
from keras.utils import to_categorical
from keras import metrics
from process_data import DataProvider
from loss_func import loss_fn, ensemble_loss_fn
import numpy as np
#import matplotlib.pyplot as plt
from keras.callbacks import CSVLogger
from keras.engine.topology import Layer
import tensorflow as tf

#create a callback for saving very large model in keras by using numpy.save
class SaveLargeModel(Callback):

    def __init__(self, filepath):
        super(SaveLargeModel, self).__init__()
        self.filepath = filepath

    def on_epoch_end(self, epoch, logs=None):
        filepath = self.filepath.format(epoch=epoch + 1)
        weights = self.model.get_weights()
        np.save(filepath, weights)

class ScaledLayer(Layer): # a scaled layer
    def __init__(self, **kwargs):
        super(ScaledLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.output_dim = input_shape[1] 
        self.W = self.add_weight(name='scale_var', shape=(1, ), initializer='one', trainable=True)
        super(ScaledLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x, mask=None):
        return tf.multiply(x, self.W)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.output_dim)


def NASNet_large_FCN(input_image, weights=None):
    """
        return Model instance
    """
    input_tensor = Input(shape=(input_image))   
    model = NASNetLarge(input_shape=input_image, input_tensor=input_tensor, include_top=False, weights=weights)
    #print model.summary()
    #return
    #reduce_stem_1 = model.get_layer()

    normal_18 = model.get_layer(name='normal_concat_18').output
    
    activation_normal_concat_7 = model.get_layer(name='activation_118').output
    activation_normal_concat_8 = model.get_layer(name='activation_130').output
    activation_normal_concat_9 = model.get_layer(name='activation_142').output
    activation_normal_concat_10 = model.get_layer(name='activation_154').output
    activation_normal_concat_11 = model.get_layer(name='activation_166').output
    activation_normal_concat_12 = model.get_layer(name='activation_178').output
    """
    activation_normal_concat_7 = ScaledLayer()(activation_normal_concat_7)
    activation_normal_concat_8 = ScaledLayer()(activation_normal_concat_8)
    activation_normal_concat_9 = ScaledLayer()(activation_normal_concat_9)
    activation_normal_concat_10 = ScaledLayer()(activation_normal_concat_10)
    activation_normal_concat_11 = ScaledLayer()(activation_normal_concat_11)
    activation_normal_concat_12 = ScaledLayer()(activation_normal_concat_12)
    """
    fuse_activation_7_10 = Add()([activation_normal_concat_7, activation_normal_concat_8, activation_normal_concat_9, \
                                activation_normal_concat_10, activation_normal_concat_11, activation_normal_concat_12])
    
    activation_normal_concat_0 = model.get_layer(name='activation_35').output
    activation_normal_concat_1 = model.get_layer(name='activation_47').output
    activation_normal_concat_2 = model.get_layer(name='activation_59').output
    activation_normal_concat_3 = model.get_layer(name='activation_71').output
    activation_normal_concat_4 = model.get_layer(name='activation_83').output
    activation_normal_concat_5 = model.get_layer(name='activation_95').output
    """
    activation_normal_concat_0 = ScaledLayer()(activation_normal_concat_0)
    activation_normal_concat_1 = ScaledLayer()(activation_normal_concat_1)
    activation_normal_concat_2 = ScaledLayer()(activation_normal_concat_2)
    activation_normal_concat_3 = ScaledLayer()(activation_normal_concat_3)
    activation_normal_concat_4 = ScaledLayer()(activation_normal_concat_4)
    activation_normal_concat_5 = ScaledLayer()(activation_normal_concat_5)
    """

    fuse_activation_0_5 = Add()([activation_normal_concat_0, activation_normal_concat_1, activation_normal_concat_2, \
                                activation_normal_concat_3, activation_normal_concat_4, activation_normal_concat_5])
    

    conv_normal_18 = Conv2D(filters=6, kernel_size=(1, 1))(normal_18)
    upscore_normal_18 = Conv2DTranspose(filters=6, kernel_size=(4, 4), strides=(2, 2), padding='same')(conv_normal_18)

    conv_fuse_7_10 = Conv2D(filters=6, kernel_size=(1, 1))(fuse_activation_7_10)
    conv_fuse_7_10 = Add()([conv_fuse_7_10, upscore_normal_18])
    upscore_fuse_7_10 = Conv2DTranspose(filters=6, kernel_size=(4, 4), strides=(2, 2), padding='same')(conv_fuse_7_10)

    conv_fuse_0_5 = Conv2D(filters=6, kernel_size=(1, 1))(fuse_activation_0_5)
    conv_fuse_0_5 = Add()([conv_fuse_0_5, upscore_fuse_7_10])
    upscore = Conv2DTranspose(filters=6, kernel_size=(16, 16), strides=(8, 8), padding='same')(conv_fuse_0_5)

    model = Model(inputs=input_tensor, outputs=upscore)
    print model.summary()
    return model

def NASNet_mobile_FCN(input_image, weights=None, fine_tune=False):
    """
        return Model instance
    """
    input_tensor = Input(shape=(input_image))   
    model = NASNetMobile(input_shape=(224, 224, 3), input_tensor=input_tensor, include_top=True, weights=weights)
    #print model.summary()
    #return
    #For fine-tuning
    if fine_tune:
        nasnet_layers = model.layers
        for layer in nasnet_layers:
            print layer
            layer.trainable = False

    stem_2 = model.get_layer(name='reduction_concat_stem_2').output
    reduce_4 = model.get_layer(name='reduction_concat_reduce_4').output
    normal_12 = model.get_layer(name='normal_concat_12').output

    conv_normal_12 = Conv2D(filters=6, kernel_size=(1, 1))(normal_12)
    upscore_normal_12 = Conv2DTranspose(filters=6, kernel_size=(4, 4), strides=(2, 2), padding='same')(conv_normal_12)

    conv_reduce_4 = Conv2D(filters=6, kernel_size=(1, 1))(reduce_4)
    fuse_4        = Add()([conv_reduce_4, upscore_normal_12])
    upscore_fuse_4 = Conv2DTranspose(filters=6, kernel_size=(4, 4), strides=(2, 2), padding='same')(fuse_4)

    conv_stem_2 = Conv2D(filters=6, kernel_size=(1, 1))(stem_2)
    fuse_2      = Add()([conv_stem_2, upscore_fuse_4])
    upscore_fuse_2 = Conv2DTranspose(filters=6, kernel_size=(16, 16), strides=(8, 8), padding='same')(fuse_2)

    model     = Model(inputs=input_tensor, outputs=upscore_fuse_2)
    print model.summary()
    
    return model

def NASNet_ensemble_FCN(input_image, weights=None, fine_tune=False):

    input_tensor = Input(shape=(input_image))   
    model = NASNetMobile(input_shape=(224, 224, 3), input_tensor=input_tensor, include_top=False, weights=weights)
    #print model.summary()
    #return
    #For fine-tuning
    if fine_tune:
        nasnet_layers = model.layers
        for layer in nasnet_layers:
            print layer
            layer.trainable = False

    stem_2 = model.get_layer(name='reduction_concat_stem_2').output
    reduce_4 = model.get_layer(name='reduction_concat_reduce_4').output
    normal_12 = model.get_layer(name='normal_concat_12').output

    #fcn_8s
    conv_normal_12 = Conv2D(filters=6, kernel_size=(1, 1))(normal_12)
    upscore_normal_12 = Conv2DTranspose(filters=6, kernel_size=(4, 4), strides=(2, 2), padding='same')(conv_normal_12)

    conv_reduce_4 = Conv2D(filters=6, kernel_size=(1, 1))(reduce_4)
    fuse_4        = Add()([conv_reduce_4, upscore_normal_12])
    upscore_fuse_4 = Conv2DTranspose(filters=6, kernel_size=(4, 4), strides=(2, 2), padding='same')(fuse_4)
    
    conv_stem_2 = Conv2D(filters=6, kernel_size=(1, 1))(stem_2)
    fuse_2      = Add()([conv_stem_2, upscore_fuse_4])
    
    output_8 = Conv2DTranspose(filters=6, kernel_size=(16, 16), strides=(8, 8), padding='same')(fuse_2)
    
    output_16 = Conv2DTranspose(filters=6, kernel_size=(32, 32), strides=(16, 16), padding='same')(fuse_4)
    
    #fcn_32s 
    output_32 = Conv2DTranspose(filters=6, kernel_size=(64, 64), strides=(32, 32), padding='same')(conv_normal_12)
    model     = Model(inputs=input_tensor, outputs=[output_8, output_16, output_32])
    print model.summary()
    return model

def NASNet_large_ensemble_FCN(input_image, weights=None, fine_tune=False):
    
    input_tensor = Input(shape=(input_image))   
    model = NASNetLarge(input_shape=input_image, input_tensor=input_tensor, include_top=False, weights=weights)
    #print model.summary()
    #reduce_stem_1 = model.get_layer()

    normal_18 = model.get_layer(name='normal_concat_18').output
    
    activation_normal_concat_7 = model.get_layer(name='activation_118').output
    activation_normal_concat_8 = model.get_layer(name='activation_130').output
    activation_normal_concat_9 = model.get_layer(name='activation_142').output
    activation_normal_concat_10 = model.get_layer(name='activation_154').output
    activation_normal_concat_11 = model.get_layer(name='activation_166').output
    activation_normal_concat_12 = model.get_layer(name='activation_178').output
    
    fuse_activation_7_10 = Add()([activation_normal_concat_7, activation_normal_concat_8, activation_normal_concat_9, \
                                activation_normal_concat_10, activation_normal_concat_11, activation_normal_concat_12])
    
    activation_normal_concat_0 = model.get_layer(name='activation_35').output
    activation_normal_concat_1 = model.get_layer(name='activation_47').output
    activation_normal_concat_2 = model.get_layer(name='activation_59').output
    activation_normal_concat_3 = model.get_layer(name='activation_71').output
    activation_normal_concat_4 = model.get_layer(name='activation_83').output
    activation_normal_concat_5 = model.get_layer(name='activation_95').output

    fuse_activation_0_5 = Add()([activation_normal_concat_0, activation_normal_concat_1, activation_normal_concat_2, \
                                activation_normal_concat_3, activation_normal_concat_4, activation_normal_concat_5])
    

    conv_normal_18 = Conv2D(filters=6, kernel_size=(1, 1))(normal_18)
    upscore_normal_18 = Conv2DTranspose(filters=6, kernel_size=(4, 4), strides=(2, 2), padding='same')(conv_normal_18)

    conv_fuse_7_10 = Conv2D(filters=6, kernel_size=(1, 1))(fuse_activation_7_10)
    conv_fuse_7_10 = Add()([conv_fuse_7_10, upscore_normal_18])
    upscore_fuse_7_10 = Conv2DTranspose(filters=6, kernel_size=(4, 4), strides=(2, 2), padding='same')(conv_fuse_7_10)

    conv_fuse_0_5 = Conv2D(filters=6, kernel_size=(1, 1))(fuse_activation_0_5)
    conv_fuse_0_5 = Add()([conv_fuse_0_5, upscore_fuse_7_10])
    output_8 = Conv2DTranspose(filters=6, kernel_size=(16, 16), strides=(8, 8), padding='same')(conv_fuse_0_5)
    output_16 = Conv2DTranspose(filters=6, kernel_size=(32, 32), strides=(16, 16), padding='same')(conv_fuse_7_10)
    output_32 = Conv2DTranspose(filters=6, kernel_size=(64, 64), strides=(32, 32), padding='same')(conv_normal_18)
    model = Model(inputs=input_tensor, outputs=[output_8, output_16, output_32])

    #print model.summary()
    return model


def train(model, learning_rate, save_model_path, logger_path, num_epochs, init_epoch, weight_path=None):
    num_classes = 6
    #image_sz= 224
    #for nasnet_large_ensemble
    image_sz = 256
    input_image = [image_sz, image_sz, 3]
    #optimizer =  SGD(lr=0.15, decay=0.97)
    optimizer = Adam(lr=learning_rate)
    #model = NASNet_mobile_FCN(input_image, weights=None, fine_tune=False)
    #model = NASNet_large_FCN(input_image)
    model.compile(optimizer=optimizer, loss=loss_fn)
    
    csv_logger = CSVLogger(logger_path, append=True, separator=';')

    #csv_logger = CSVLogger('checkpoint/NASNet_large_FCN_weighted_fuse/log_NASNet_large_FCN_weighted_fuse.csv', append=True, separator=';')
    #load weight for continue training 
    #model.load_weights('checkpoint/NASNet_mobile_FCN/weights.50.hdf5')
    
    if init_epoch != 0 and weight_path != None:
        print "Init_epoch greater than 0 need specify weight_path"
        return

    if weight_path != None:
        weights = np.load(weight_path)
        model.set_weights(weights)

    #model_checkpoint = ModelCheckpoint('checkpoint/NASNet_mobile_FCN/weights.{epoch:02d}.hdf5', save_weights_only=True)
    #model_checkpoint = SaveLargeModel('checkpoint/NASNet_large_FCN_weighted_fuse/weights.{epoch:02d}.npy')

    model_checkpoint = SaveLargeModel(save_model_path)
    X_train, Y_train, X_valid, Y_valid = DataProvider("./ISPRS_semantic_labeling_Vaihingen").load_data(images_from_each=1000, image_size=image_sz,ground_truth=True,take_all=1 )
    #hist = model.fit(x=X_train, y=Y_train, batch_size=2, callbacks=[model_checkpoint], validation_data=(X_valid, Y_valid), epochs=10, verbose=2, shuffle=True)
    hist = model.fit(x=X_train, y=[Y_train, Y_train, Y_train], validation_data =(X_valid, [Y_valid, Y_valid, Y_valid]), batch_size=10, callbacks=[csv_logger, model_checkpoint], epochs=num_epochs,initial_epoch=init_epoch, verbose=2, shuffle=True )
    
    """
    plt.plot(hist.history['loss'])  
    plt.title('model loss')  
    plt.ylabel('loss')  
    plt.xlabel('epoch')  
    plt.legend(['train', 'test'], loc='upper left')  
    #plt.show()  
    plt.savefig('loss_graph.png')
    #with open('logs/NASNet_FCN/logs.txt','w') as f:
    #    f.write(hist.history)
    """
    #generate result 
    
    """
    overlap_size = 200  
    data_provider = DataProvider('./ISPRS_semantic_labeling_Vaihingen')
    for idx in DataProvider.test_idx:
        test_data,test_data_info = data_provider.get_chunk_data(idx, overlap_size=overlap_size)
        print "Load test data success"

        print test_data.shape
        preds = model.predict(test_data, batch_size=1)
        data_provider.merge_chunks(idx, np.array(preds), test_data_info, return_softmax=False)


    data_provider = DataProvider('./ISPRS_semantic_labeling_Vaihingen')
    for idx in DataProvider.labeled_idx:
        test_data,test_data_info = data_provider.get_chunk_data(idx, overlap_size=overlap_size)
        print "Load test data success"

        print test_data.shape
        preds = model.predict(test_data, batch_size=1)
        data_provider.merge_chunks(idx, np.array(preds), test_data_info, return_softmax=False)
    """

if __name__ == '__main__':
    model = NASNet_large_FCN((224, 224, 3))
    #model = NASNet_mobile_FCN((224, 224, 3))
    #model = NASNet_ensemble_FCN((224, 224, 3), weights='imagenet', fine_tune=True)
    #model = NASNet_large_ensemble_FCN((256, 256, 3))
    #train(model, 0.0001, 'checkpoint/NASNet_large_ensemble_FCN/weights.{epoch:02d}.npy','checkpoint/NASNet_large_ensemble_FCN/log_NASNet_mobile_ensemble_FCN.csv', num_epochs=150, init_epoch=0)
    #NASNet_mobile_FCN([224, 224, 3], weights='imagenet', fine_tune=True)
    #NASNet_large_FCN([224, 224 , 3])