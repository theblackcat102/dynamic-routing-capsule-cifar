
from models.capsule_net import train as capsule_train
from models.capsule_net import test as capsule_test
from models.resnet import train as resnet_train
from utils.helper_function import plot_log
from sys import argv

if __name__ == "__main__":
    # # resnet_train()
    epochs=argv[1]
    try:
        batch_size=argv[3]
        dataset=argv[2]
    except:
        batch_size=128
        dataset=argv[2]
    capsule_train(epochs,batch_size,dataset)
    capsule_test(epochs,dataset)
    #plot_log("results/resnet-cifar-10-log.csv")
