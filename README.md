# dynamic-routing-capsule-cifar
reference from : https://github.com/XifengGuo/CapsNet-Keras

# Testing Cifar-10 datasets using Dynamic Routing between Capsules
This data will be compared with resnet18(18 layers deep) which contain a similar parameters with the tested capsule net in Hinton's paper.

Each epoch only takes less than 5 minutes in resnet, so I left it to train for over 100 epoch, which checkpoint stops at epoch 44
![Resnet](https://github.com/theblackcat102/dynamic-routing-capsule-cifar/blob/master/results/Figure_2.png)

Each epoch in capsule net takes about 30 minutes, in this case, I only left it training for 60 epoch. 
![Capsule](https://github.com/theblackcat102/dynamic-routing-capsule-cifar/blob/master/results/Figure_1.png)

#### In the end, resnet yield an accuracy of 0.8231, while the "simple" capsule net only 0.7307
Here is the reconstruction result from the capsule 3 fully connected decoder. I suspect the bad reconstruction was due to simple FC network rather than using Image GAN structure of stacking Maxpooling(oops) and convolution layer. According to the paper, this addition reconstruction section contribute quite a significant increase in the final accuracy. In my opinion, it really doesn't makes sense, since the major study here was to evaluate the performance of capsule structure.

![Reconstruction](https://github.com/theblackcat102/dynamic-routing-capsule-cifar/blob/master/results/real_and_recon.png)
