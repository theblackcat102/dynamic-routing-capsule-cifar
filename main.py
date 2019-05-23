
from models.capsule_net import train as capsule_train
from models.capsule_net import test as capsule_test
from models.resnet import train as resnet_train
from utils.helper_function import plot_log
from sys import argv

import argparse
arguments = argparse.ArgumentParser()
arguments.add_argument('--epocs',default=100)
arguments.add_argument('--batch_size',default=128)
arguments.add_argument('--dataset',default=1)
arguments.add_argument('--is_relu',default=1)
arguments.add_argument('--has',default=1)
arguments.add_argument('--normalize',default=0)

if __name__ == "__main__":
    arg=arguments.parse_args()
    # # resnet_train()
    epochs=arg.epocs
    is_relu=bool(arg.is_relu)
    print(is_relu)
    batch_size=int(arg.batch_size)
    dataset=arg.dataset
    has=bool(arg.has)
    normalize=bool(arg.normalize)
    capsule_train(epochs=epochs,
                  batch_size=batch_size,
                  mode=dataset,
                  is_relu=is_relu,
                  has=has,
                  normal=normalize)
    if(int(dataset)==1):
        maske='Cifar10'
    else:
        maske='KTH'
    if(normal==False):
        if(has):
            if(is_relu):
                best_model_name=maske+'_Relu.h5'
            else:
                best_model_name=maske+'_Leaky_Relu.h5'
                pass
        else:
            best_model_name=maske+'.h5'
    if(normal==True):
        if(has):
            if(is_relu):
                best_model_name=maske+'_Relu_norm.h5'
            else:
                best_model_name=maske+'_Leaky_Relu_norm.h5'
                pass
        else:
            best_model_name=maske+'_norm.h5'
    capsule_test(epochs,dataset,
                 normal=normalize,
                 best_model_name=best_model_name)
    #plot_log("results/resnet-cifar-10-log.csv")
 