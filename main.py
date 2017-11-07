from models.capsule_net import train as capsule_train
from models.capsule_net import test as capsule_test

if __name__ == "__main__":
    capsule_train(150, 16, 0)
    # capsule_test(19)
