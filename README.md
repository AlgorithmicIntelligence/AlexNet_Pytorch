I try to implement the code from the paper, but it didn't work as expect, so I modified some of it, like learning rate, number of classes, PCA augmentation, etc... But most of all are really similar with the paper.

The result is training by ILSVRC for 100 classes, I trained with lr*1, lr*0.1, lr*0.01. Obviously, it is overfitting when learning rate is small.

When lr*1:

<img src="https://github.com/AlgorithmicIntelligence/AlexNet_Pytorch/blob/master/README/Accuracy.png" width="450"><img src="https://github.com/AlgorithmicIntelligence/AlexNet_Pytorch/blob/master/README/Loss.png" width="450">


When lr*0.1:

<img src="https://github.com/AlgorithmicIntelligence/AlexNet_Pytorch/blob/master/README/Accuracy0.1.png" width="450"><img src="https://github.com/AlgorithmicIntelligence/AlexNet_Pytorch/blob/master/README/Loss0.1.png" width="450">

When lr*0.01:

<img src="https://github.com/AlgorithmicIntelligence/AlexNet_Pytorch/blob/master/README/Accuracy0.01.png" width="450"><img src="https://github.com/AlgorithmicIntelligence/AlexNet_Pytorch/blob/master/README/Loss0.01.png" width="450">
