
from models.capsule_net import train as capsule_train
from models.capsule_net import test as capsule_test
from models.resnet import train as resnet_train
from utils.helper_function import plot_log

if __name__ == "__main__":
    # # resnet_train()
    capsule_train(80,128, 1)
    #capsule_test(61)
    #plot_log("results/resnet-cifar-10-log.csv")
