from models.capsule_net import train as capsule_train
from models.capsule_net import test as capsule_test
from models.resnet import train as resnet_train

if __name__ == "__main__":
    resnet_train()
    # capsule_train(150, 16, 1)
    # capsule_test(19)
