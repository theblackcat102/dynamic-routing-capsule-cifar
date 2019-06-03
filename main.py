from models.capsule_net import train as capsule_train
from models.capsule_net import test as capsule_test
from models.capsule_net import test_model
#from models.resnet import train as resnet_train
#from utils.helper_function import plot_log
import os

from sys import argv,exit

import argparse
arguments = argparse.ArgumentParser()
arguments.add_argument('--epocs',default=1,type=int)
arguments.add_argument('--batch_size',default=128,type=int)
arguments.add_argument('--dataset',default=1,type=int)
arguments.add_argument('--is_relu',default=1,type=int)
arguments.add_argument('--has',default=1,type=int)
arguments.add_argument('--version',default='')
arguments.add_argument('--lear_rate',default=0.01)


arguments.add_argument('--large_test',default=0,type=int)

arguments.add_argument('--model_path',default='fakemodel.h5')
arguments.add_argument('--dataset_path',default='fakepath')
arguments.add_argument('--save_path',default='.')

def save_best_model(mode=None,
                    version=None,
                    best_model_name=None):
    if(mode==1):
        maske='Cifar10'
    else:
        maske='KTH'
        pass
    path='weights'+maske
    print(path)
    best_ecpoch=int(max([x.split('.')[0].split('-')[-1] for x in os.listdir(path)]))
    best_model_path='weights'+maske+'/capsule-net-'+str(10)+'weights-{:02d}.h5'.format(best_ecpoch)
    print(best_model_path)
    os.rename(src=best_model_path,dst='modelsKTH/'+best_model_name)
    pass

if __name__ == "__main__":
    arg=arguments.parse_args()
    # # resnet_train()
    epochs=arg.epocs
    is_relu=bool(arg.is_relu)
    lear_rate=float(arg.lear_rate)
    batch_size=int(arg.batch_size)
    dataset=arg.dataset
    has=bool(arg.has)
    version=str(arg.version)
    test=int(arg.large_test)
    model_dir=arg.model_path
    dataset_path=arg.dataset_path
    save_path=arg.save_path
    if(test==0):
        capsule_train(epochs=epochs,
                      batch_size=batch_size,
                      mode=dataset,
                      is_relu=is_relu,
                      has=has,
                      version=version,
                      lear=lear_rate)
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
        print(best_model_name)
        
        capsule_test(epoch=epochs,
                     batch_size=batch_size,
                     mode=dataset,
                     version=version,
                     best_model_name=best_model_name)
        #plot_log("results/resnet-cifar-10-log.csv")
        '''
        
        save_best_model(mode=dataset,
                        version=version,
                        best_model_name=best_model_name)
        '''
    if(test==1):
        img=dataset_path+'/'+'KTH-b-Test.npy'
        lab=dataset_path+'/'+'KTH-b-Test-lab.npy'
        dataset={'Images':img,'Labels':lab}
        test_model(model_path=model_dir,
                   dataset_path=dataset,
                   save_path=save_path)