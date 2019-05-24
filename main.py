
from models.capsule_net import train as capsule_train
from models.capsule_net import test as capsule_test
from models.resnet import train as resnet_train
from utils.helper_function import plot_log
from sys import argv

import argparse
arguments = argparse.ArgumentParser()
arguments.add_argument('--epocs',default=1)
arguments.add_argument('--batch_size',default=128)
arguments.add_argument('--dataset',default=1)
arguments.add_argument('--is_relu',default=1)
arguments.add_argument('--has',default=1)
arguments.add_argument('--version',default='')

if __name__ == "__main__":
    arg=arguments.parse_args()
    # # resnet_train()
    epochs=arg.epocs
    is_relu=bool(arg.is_relu)
    print(is_relu)
    batch_size=int(arg.batch_size)
    dataset=arg.dataset
    has=bool(arg.has)
    version=str(arg.version)
    capsule_train(epochs=epochs,
                  batch_size=batch_size,
                  mode=dataset,
                  is_relu=is_relu,
                  has=has,
                  version=version)
    if(int(dataset)==1):
        maske='Cifar10'
    else:
        maske='KTH'
        pass
    if(has):
        if(is_relu):
            best_model_name=maske+version+'_Relu.h5'
        else:
            best_model_name=maske+version+'_Leaky_Relu.h5'
            pass
    else:
        best_model_name=maske+version+'.h5'
    capsule_test(epochs,dataset,
                 version=version,
                 best_model_name=best_model_name)
    #plot_log("results/resnet-cifar-10-log.csv")
 