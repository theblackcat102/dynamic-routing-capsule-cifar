# dynamic-routing-capsule-cifar
reference from : https://github.com/XifengGuo/CapsNet-Keras

# Testing Cifar-10 datasets using Dynamic Routing between Capsules
This data will be compared with resnet18 which contain a similar parameters 

# I first started with a resnet 18 layer which has a similar amount of parameters with capsule net
Each epoch only takes less than 5 minutes, so I left it to train for over 100 epoch, which checkpoint stops at epoch 44

![Resnet](https://github.com/theblackcat102/dynamic-routing-capsule-cifar/blob/master/results/Figure_2.png)

Each epoch in capsule net takes about 30 minutes, in this case, I only left it training for 60 epoch. 
![Capsule](https://github.com/theblackcat102/dynamic-routing-capsule-cifar/blob/master/results/Figure_1.png)

### In the end, resnet yield an accuracy of 0.8231, while the "simple" capsule net only 0.7307
