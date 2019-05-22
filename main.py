
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
arguments.add_argument('--isrelu',default=1)

if __name__ == "__main__":
    arg=arguments.parse_args()
    # # resnet_train()
    epochs=arg.epocs
    isrelu=bool(arg.isrelu)
    print(isrelu)
    batch_size=arg.batch_size
    dataset=arg.dataset
    capsule_train(epochs,batch_size,dataset,isrelu)
    capsule_test(epochs,dataset)
    #plot_log("results/resnet-cifar-10-log.csv")
 