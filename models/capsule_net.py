import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.layers.advanced_activations import LeakyReLU


config = tf.ConfigProto()
config.gpu_options.allow_growth = False
config.gpu_options.per_process_gpu_memory_fraction =0.8
set_session(tf.Session(config=config))
import sys
from sklearn.metrics import confusion_matrix

from keras.layers import (
    Input,
    Conv2D,
    Activation,
    Dense,
    Flatten,
    Reshape,
    Dropout
)
from keras.layers.merge import add
from keras.regularizers import l2
from keras.models import Model
from models.capsule_layers import CapsuleLayer, PrimaryCapsule, Length,Mask
from keras.layers.normalization import BatchNormalization
import keras.backend as K
from keras import optimizers
from utils.helper_function import load_cifar_10,load_cifar_100
from models.capsulenet import CapsNet as CapsNetv1
import numpy as np
import time as t
import os

def timed(time):
    time=t.strftime('%H:%M:%S', t.gmtime(time))+str(time%1)[1:4]
    print('tiempo de entrenamiento')
    print(time)


def convolution_block(input,kernel_size=8,filters=16,kernel_regularizer=l2(1.e-4)):
    conv2 = Conv2D(filters=filters,kernel_size=kernel_size,kernel_regularizer=kernel_regularizer,
                    kernel_initializer="he_normal",padding="same")(input)
    norm = BatchNormalization(axis=3)(conv2)
    activation = Activation("relu")(norm)
    return activation    

def CapsNet(input_shape,n_class,n_route,n_prime_caps=32,dense_size = (512,1024),is_relu=True):
    conv_filter = 256
    n_kernel = 24
    primary_channel =64
    primary_vector = 9
    vector_dim = 9

    target_shape = input_shape

    input = Input(shape=input_shape)

    # TODO: try leaky relu next time
    conv1 = Conv2D(filters=conv_filter,kernel_size=n_kernel, strides=1, padding='valid', activation='relu',name='conv1',kernel_initializer="he_normal")(input)

    primary_cap = PrimaryCapsule(conv1,dim_vector=8, n_channels=64,kernel_size=9,strides=2,padding='valid')

    routing_layer = CapsuleLayer(num_capsule=n_class, dim_vector=vector_dim, num_routing=n_route,name='routing_layer')(primary_cap)

    output = Length(name='output')(routing_layer)

    y = Input(shape=(n_class,))
    masked = Mask()([routing_layer,y])
    
    x_recon = Dense(dense_size[0])(masked)
    if(is_relu):
        x_recon=Activation('relu')(x_recon)
    else:
        x_recon=LeakyReLU(0.2)(x_recon)

    for i in range(1,len(dense_size)):
        x_recon = Dense(dense_size[i])(x_recon)
        if(is_relu):
            x_recon=Activation('relu')(x_recon)
        else:
            x_recon=LeakyReLU(0.2)(x_recon)
    # Is there any other way to do  
    x_recon = Dense(target_shape[0]*target_shape[1]*target_shape[2])(x_recon)
    if(is_relu):
        x_recon=Activation('relu')(x_recon)
    else:
        x_recon=LeakyReLU(0.2)(x_recon)
    x_recon = Reshape(target_shape=target_shape,name='output_recon')(x_recon)
    x_recon=tf.layers.conv2d_transpose(x_recon,kernel_size=5,strides=3,filters=64)
    if(is_relu):
        x_recon=Activation('relu')(x_recon)
    else:
        x_recon=LeakyReLU(0.2)(x_recon)
    x_recon=tf.layers.conv2d_transpose(x_recon,kernel_size=5,strides=3,filters=256)
    if(is_relu):
        x_recon=Activation('relu')(x_recon)
    else:
        x_recon=LeakyReLU(0.2)(x_recon)
    x_recon=tf.layers.conv2d_transpose(x_recon,kernel_size=6,strides=2,filters=3)
    if(is_relu):
        x_recon=Activation('relu')(x_recon)
    else:
        x_recon=LeakyReLU(0.2)(x_recon)
    return Model([input,y],[output,x_recon])

# why using 512, 1024 Maybe to mimic original 10M params?
def CapsNetv2(input_shape,n_class,n_route,n_prime_caps=32,dense_size = (512,1024)):
    conv_filter = 64
    n_kernel = 16
    primary_channel =64
    primary_vector = 12
    capsule_dim_size = 8

    target_shape = input_shape

    input = Input(shape=input_shape)

    # TODO: try leaky relu next time
    conv_block_1 = convolution_block(input,kernel_size=16,filters=64)
    primary_cap = PrimaryCapsule(conv_block_1,dim_vector=capsule_dim_size,n_channels=primary_channel,kernel_size=9,strides=2,padding='valid')    
    # Suppose this act like a max pooling 
    routing_layer = CapsuleLayer(num_capsule=n_class,dim_vector=capsule_dim_size*2,num_routing=n_route,name='routing_layer_1')(primary_cap)
    output = Length(name='output')(routing_layer)

    y = Input(shape=(n_class,))
    masked = Mask()([routing_layer,y])
    
    x_recon = Dense(dense_size[0],activation='relu')(masked)

    for i in range(1,len(dense_size)):
        x_recon = Dense(dense_size[i],activation='relu')(x_recon)
    # Is there any other way to do  
    x_recon = Dense(np.prod(target_shape),activation='relu')(x_recon)
    x_recon = Reshape(target_shape=target_shape,name='output_recon')(x_recon)
    print(x_recon)
    sys.exit(1)
    # conv_block_2 = convolution_block(routing_layer)
    # b12_sum = add([conv_block_2,conv_block_1])

    return Model([input,y],[output,x_recon])

def margin_loss(y_true, y_pred):
    """
    Margin loss for Eq.(4). When y_true[i, :] contains not just one `1`, this loss should work too. Not test it.
    :param y_true: [None, n_classes]
    :param y_pred: [None, num_capsule]
    :return: a scalar loss value.
    """
    L = y_true * K.square(K.maximum(0., 0.9 - y_pred)) + \
        0.5 * (1 - y_true) * K.square(K.maximum(0., y_pred - 0.1))

    return K.mean(K.sum(L, 1))

def train(epochs,batch_size,mode,is_relu,has=True,version='',lear=0.01):
    mode=int(mode)
    if(mode==1):
        maske='Cifar10'
    else:
        maske='KTH'
    print(maske)
    print(len(os.listdir('weights'+maske+'/')))
    if(len(os.listdir('weights'+maske+'/'))>2):
        print('already trained')
        return 0
    from keras import callbacks
    from keras.utils.vis_utils import plot_model
    if mode==1:
        num_classes = 10
        (x_train,y_train),(x_test,y_test) = load_cifar_10()
        model = CapsNetv1(input_shape=[32,32, 3],
                            n_class=num_classes,
                            n_route=3)
        print('x_train shape:', x_train.shape)
        print(x_train.shape[0], 'train samples')
        print(x_test.shape[0], 'test samples')
    else:
        num_classes = 10
        try:
            x_train=np.load('/home/vision/Documentos/kth_img_train'+version+'.npy')
            y_train=np.load('/home/vision/Documentos/kth_lab_train'+version+'.npy')
            with tf.Session() as sess:
                y_train=sess.run(tf.one_hot(y_train,10))
        except:
            print('kth no cargado')
        try:
            x_test=np.load('/home/vision/Documentos/kth_img_test'+version+'.npy')
            y_test=np.load('/home/vision/Documentos/kth_lab_test'+version+'.npy')
            with tf.Session() as sess:
                y_test=sess.run(tf.one_hot(y_test,10))
            print('x_train shape:', x_train.shape)
            print(x_train.shape[0], 'train samples')
            print(x_test.shape[0], 'test samples')
        except:
            print('kth no cargado')
        model = CapsNetv1(input_shape=[200,200, 3],
                            n_class=num_classes,
                            n_route=3,kth=True,
                            is_relu=is_relu,
                            has=has)

    model.summary()
    log = callbacks.CSVLogger('results'+maske+'/capsule-net-'+str(num_classes)+'-log.csv')
    tb = callbacks.TensorBoard(log_dir='results'+maske+'/tensorboard-capsule-net-'+str(num_classes)+'-logs',
                               batch_size=batch_size, histogram_freq=True)
    checkpoint = callbacks.ModelCheckpoint('weights'+maske+'/capsule-net-'+str(num_classes)+'weights-{epoch:02d}.h5',
                                           save_best_only=False, save_weights_only=True, verbose=1)
    lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: 0.001 * np.exp(-epoch / 10.))

    plot_model(model, to_file='models'+maske+'/capsule-net-'+str(num_classes)+'.png', show_shapes=True)

    model.compile(optimizer=optimizers.Adam(lr=lear),
                  loss=[margin_loss, 'mse'],
                  loss_weights=[1., 0.1],
                  metrics={'output_recon':'accuracy','output':'accuracy'})
    from utils.helper_function import data_generator

    generator = data_generator(x_train,y_train,batch_size)
    # Image generator significantly increase the accuracy and reduce validation loss
    start=t.time()
    model.fit_generator(generator,
                        steps_per_epoch=x_train.shape[0] // batch_size,
                        validation_data=([x_test, y_test], [y_test, x_test]),
                        epochs=int(epochs), verbose=1, max_q_size=100,
                        callbacks=[log,tb,checkpoint,lr_decay])
    stop=t.time()
    timed(stop-start)

def test(epoch, mode=1,version='',best_model_name='.'):
    epoch=int(epoch)
    from PIL import Image
    from utils.helper_function import combine_images
    if(mode==1):
        maske='Cifar10'
    else:
        maske='KTH'
    if mode==1:
        num_classes = 10
        (x_train,y_train),(x_test,y_test) = load_cifar_10()
        model = CapsNetv1(input_shape=[32,32, 3],
                            n_class=num_classes,
                            n_route=3)
        print('x_train shape:', x_train.shape)
        print(x_train.shape[0], 'train samples')
        print(x_test.shape[0], 'test samples')
    else:
        num_classes = 10
        try:
            x_train=np.load('/home/vision/Documentos/kth_img_train'+version+'.npy')
            y_train=np.load('/home/vision/Documentos/kth_lab_train'+version+'.npy')
            with tf.Session() as sess:
                y_train=sess.run(tf.one_hot(y_train,10))
        except:
            print('kth no cargado')
        try:
            x_test=np.load('/home/vision/Documentos/kth_img_test'+version+'.npy')
            y_test=np.load('/home/vision/Documentos/kth_lab_test'+version+'.npy')
            with tf.Session() as sess:
                y_test=sess.run(tf.one_hot(y_test,10))
            print('x_train shape:', x_train.shape)
            print(x_train.shape[0], 'train samples')
            print(x_test.shape[0], 'test samples')
        except:
            print('kth no cargado')
        model = CapsNetv1(input_shape=[200,200, 3],
                            n_class=num_classes,
                            n_route=3,kth=True)
        pass
    accuracy=[]
    print(maske)
    print(mode)
    conf_matrix=[]
    epochs=epoch
    for epoch in range(1,epochs+1):
        try:
            model_path='weights'+maske+'/capsule-net-'+str(num_classes)+'weights-{:02d}.h5'.format(epoch)
            model.load_weights(model_path)
            print('model path '+model_path)
            print("Weights loaded, start validation con epoch "+str(epoch))
            y_pred, x_recon = model.predict([x_test, y_test], batch_size=100)
            ac=np.sum(np.argmax(y_pred, 1) == np.argmax(y_test, 1))/y_test.shape[0]
            accuracy.append(ac)
            conf_matrix.append(confusion_matrix(y_true=np.argmax(y_test, 1), y_pred=np.argmax(y_pred, 1)))
            print('Test acc:',ac)  
        except:
            print('not saver epoch '+str(epoch))
            pass
        if((epoch/epochs)>10):
            img = combine_images(np.concatenate([x_test[:50],x_recon[:50]]))
            image = img*255
            Image.fromarray(image.astype(np.uint8)).save('results'+maske+'/real_and_recon_'+str(epoch+1)+'.png')
        pass
    accuracy=np.array(accuracy)
    conf_matrix=np.array(conf_matrix)
    best_ecpoch=np.argmax(accuracy)+1
    best_model_path='weights'+maske+'/capsule-net-'+str(num_classes)+'weights-{:02d}.h5'.format(best_ecpoch)
    os.rename(src=best_model_path,
              dst='modelsKTH/'+best_model_name)
    np.save('results'+maske+'/acc',accuracy)
    np.save('results'+maske+'/cnf',conf_matrix)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    