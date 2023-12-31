# DESIGN A CNN USING DILATION CONVOLUTION AND DEPTH-WISE CONVOLUTION

## TARGET :
* The architecture must have 
<pre>
PrepLayer - 
  Conv 3x3 s1, p1 >> BN >> RELU [64k]
Layer1 -
  X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [128k]
  R1 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [128k] 
  Add(X, R1)
Layer 2 -
  Conv 3x3 [256k]
  MaxPooling2D
  BN
  ReLU
Layer 3 -
  X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [512k]
  R2 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [512k]
  Add(X, R2)
MaxPooling with Kernel Size 4
FC Layer 
SoftMax
</pre>
* Uses One Cycle Policy such that:
<pre>
  Total Epochs = 24
  Max at Epoch = 5
  LRMIN = FIND
  LRMAX = FIND
  NO Annihilation
</pre>
* Use the transform -RandomCrop 32, 32 (after padding of 4) >> FlipLR >> Followed by CutOut(8, 8)
* Batch size = 512
* Use ADAM and CrossEntropyLoss
* Target Accuracy: 90%

## CONTENTS :
- [DATASET](#dataset)
- [IMPORTING_LIBRARIES](#importing_libraries)
- [SET_THE_ALBUMENTATIONS](#set_the_albumentations)
- [DATA_AUGMENTATIONS](#data_augmentations)
- [SET_DATA_LOADER](#Set_Data_Loader)
- [CNN_MODEL](#cnn_model)
- [TRAINING_THE_MODEL](training_the_model)
- [LR_SCHEDULAR](lr_schedular)
- [RESULTS](results)


## DATASET 
### CIFAR DATASET
CIFAR-10 is an established computer-vision dataset used for object recognition. It is a subset of the 80 million tiny images dataset and consists of 60,000 32x32 color (RGB) images containing one of 10 object classes, with 6000 images per class.

## IMPORTING_LIBRARIES
Import the required libraries. 
* NumPy is used for numerical operations. The torch library is used to import Pytorch.
* Pytorch has an nn component that is used for the abstraction of machine learning operations. 
* The torchvision library is used so that to import the CIFAR-10 dataset. This library has many image datasets and is widely used for research. The transforms can be imported to resize the image to equal size for all the images. 
* The optim is used train the neural Networks.
* MATLAB libraries are imported to plot the graphs and arrange the figures with labeling
* Albumenations are imported for Middle Man's Data Augmentation Strategy
* cv2 is imported 
* torch_lr_finder is imported to find the maximum and minimum learning rate

## SET_THE_ALBUMENTATIONS
* cv2.setNumThreads(0) sets the number of threads used by OpenCV to 0. This is done to avoid a deadlock when using OpenCV’s resize method with PyTorch’s dataloader1.

* cv2.ocl.setUseOpenCL(False) disables the usage of OpenCL in OpenCV2 and is used when you want to disable the usage of OpenCL.

* The  class is inherited from torchvision.datasets.CIFAR10. It overrides the __init__ and __getitem__ methods of the parent class. The __getitem__ method returns an image and its label after transforming the image3. (This is to be done while using Albumenations)


## DATA_AUGMENTATIONS
For this import albumentations as A

Middle-Class Man's Data Augmentation Strategy is used. Like
### Normalize
<pre>
Syntax:
     A.Normalize(
            mean = (0.4914, 0.4822, 0.4465),
            std = (0.2470, 0.2435, 0.2616), always_apply = True)

Normalization is a common technique used in deep learning to scale the pixel values of an image to a standard range. This ensures that the input features have similar ranges and are centered around zero. 
Normalization is done with respect to mean and standard Deviation.
For CIFAR10 (RGB) will have 3 means and 3 standard deviation that is equal to 
(0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
Normalize all the iamges
Applied to training and test data
</pre>
### HorizontalFlip :
<pre>
Syntax : A.HorizontalFlip()
Flip the input horizontally around the y-axis.
Args:     
p (float): probability of applying the transform. Default: 0.5.
Applied only to Training data
</pre>
### ShiftScaleRotate :
<pre>
Syntax:
A.ShiftScaleRotate (shift_limit=(-0.2,0.2), scale_limit=(-0.2,0.2), rotate_limit=(-15, 15), p=0.5)
Randomly apply affine transforms: translate, scale, and rotate the input.

    Args:
        shift_limit ((float, float) or float): shift factor range for both height and width. If shift_limit
            is a single float value, the range will be (-shift_limit, shift_limit). Absolute values for lower and
            upper bounds should lie in the range [0, 1]. Default: (-0.0625, 0.0625).
        scale_limit ((float, float) or float): scaling factor range. If scale_limit is a single float value, the
            range will be (-scale_limit, scale_limit). Note that the scale_limit will be biased by 1.
            If scale_limit is a tuple, like (low, high), sampling will be done from the range (1 + low, 1 + high).
            Default: (-0.1, 0.1).
        rotate_limit ((int, int) or int): rotation range. If rotate_limit is a single int value, the
            range will be (-rotate_limit, rotate_limit). Default: (-45, 45).
        p(float): the probability of applying the transform. Default: 0.5.

Applied only to Training data
</pre>
### PadIfNeeded
<pre>
Syntax:
    A.PadIfNeeded(min_height=36, min_width=36, p=1.0),
PadIfNeeded is an image augmentation technique that pads the input image on all four sides if the side is less than the desired number. The desired number is specified by the min_height and min_width parameters. In this case, padding is equal to 4.

</pre>
### RandomCrop
<pre>
Syntax:
  A.RandomCrop(height=32, width=32, always_apply = False,p=1.0),

RandomCrop is an image augmentation technique that crops a random part of the input and rescales it to some size without loss of bounding boxes. The height and width parameters specify the size of the crop. In this case, image is cropped to size 32 X 32
</pre>
### CenterCrop
<pre>
A.CenterCrop(32, 32, always_apply=False, p=1.0)
It crops the center square of an image with a side length of 32 pixels. The always_apply parameter is set to False by default, which means that the transformation will not be applied to all images in the dataset. The p parameter is set to 1.0 by default, which means that the transformation will be applied to all images with a probability of 100%

</pre>
### Cutout
<pre>
Syntax:
 A.CoarseDropout(max_holes = 1, max_height=8, max_width=8, min_holes = 1, min_height=8, min_width=8,
                        fill_value=(0.4914, 0.4822, 0.4465), always_apply = True)
 It is similar to a cutout

    Args:
        max_holes(int): The maximum number of rectangular regions to be masked. (for CIFAR10 Dataset its 32X32)
        max_height(int): The maximum height of the rectangular regions. 
        max_width(int): The maximum width of the rectangular regions.
        min_holes(int): The minimum number of rectangular regions to be masked.
        min_height(int): The minimum height of the rectangular regions.
        min_width(int): The minimum width of the rectangular regions.
        fill_value(float): The value to be filled in the masked region. It can be a tuple or a single value. 
            It is usually equal to the mean of the dataset for CIFAR10 (0.4914, 0.4822, 0.4465)
        always_apply = True - Applies to all the images
       
Applied only to Training data 
</pre>


### ToTensorV2
<pre>
Syntax:
    ToTensorV2()

To make this function work we need to ToTensorV2 from albumentations.pytorch.transforms
It is a class in the PyTorch library that converts an image to a PyTorch tensor. It is part of the torchvision.transforms module and is used to preprocess images before feeding them into a neural network. 

Applied to training and test data
</pre>
 #### PRINTED TRAIN_TRANSFORMS and TEST_TRANSFORMS 
 <pre>
Files already downloaded and verified
Files already downloaded and verified
Compose([
  Normalize(always_apply=True, p=1.0, mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.2435, 0.2616), max_pixel_value=255.0),
  HorizontalFlip(always_apply=False, p=0.5),
  ShiftScaleRotate(always_apply=False, p=0.5, shift_limit=(-0.2, 0.2), scale_limit=(-0.19999999999999996, 0.19999999999999996), rotate_limit=(-15, 15), interpolation=1, border_mode=4, value=None, mask_value=None),
  PadIfNeeded(always_apply=False, p=1.0, min_height=36, min_width=36, border_mode=4, value=None, mask_value=None),
  RandomCrop(always_apply=False, p=1.0, height=32, width=32),
  CenterCrop(always_apply=False, p=1.0, height=32, width=32),
  CoarseDropout(always_apply=False, p=0.5, max_holes=1, max_height=8, max_width=8, min_holes=1, min_height=8, min_width=8),
  ToTensorV2(always_apply=True, p=1.0),
], p=1.0, bbox_params=None, keypoint_params=None, additional_targets={})
Compose([
  Normalize(always_apply=True, p=1.0, mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.201), max_pixel_value=255.0),
  ToTensorV2(always_apply=True, p=1.0),
], p=1.0, bbox_params=None, keypoint_params=None, additional_targets={})
</pre>

## SET_DATA_LOADER
* Batch Size = 512
* Number of Workers = 0
* CUDA is used

#### PRINTED TRAIN and TEST LOADER:
<pre>
Files already downloaded and verified
Files already downloaded and verified
<torch.utils.data.dataloader.DataLoader object at 0x7e6edbc09570>
length of train_loader 98
<torch.utils.data.dataloader.DataLoader object at 0x7e6edbc0b7c0>
length of test_loader 20

</pre>
#### SAMPLE IMAGES IN TRAIN LOADER
![Images S10/images.png](https://github.com/RajidiSahithi/Session_10/blob/main/Images%20S10/images.png)]

## CNN_MODEL

#### MODEL
<pre>
Preparation Layer:
 This Block Contains 1 Convolution layer with Kernal Size 3X3 and Stride is 1 . (Each layer has Batch Normalization, Activation Function (Relu), and Dropout)
   (conv1): Sequential(
    (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(inplace=True)
  )
</pre>
<pre>
Layer 1:
  This Block Contains 4 Convolution layers with Kernal Size 3X3 and Stride is 1 for all layers and max-pooling with stride 2 and kernel size = 2
  
  
  (conv11): Sequential(
    (0): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (3): ReLU(inplace=True)
    (4): Dropout(p=0.01, inplace=False)
  )
  (conv12): Sequential(
    (0): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(inplace=True)
    (3): Dropout(p=0.01, inplace=False)
    (4): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (5): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (6): ReLU(inplace=True)
    (7): Dropout(p=0.01, inplace=False)
  )
  </pre>
  <pre>
  Layer 2:
  (conv2): Sequential(
    (0): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (3): ReLU(inplace=True)
    (4): Dropout(p=0.01, inplace=False)
  )
  </pre>
  <pre>
  Layer 3:

  (conv31): Sequential(
    (0): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (3): ReLU(inplace=True)
    (4): Dropout(p=0.01, inplace=False)
  )
  (conv32): Sequential(
    (0): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(inplace=True)
    (3): Dropout(p=0.01, inplace=False)
    (4): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (5): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (6): ReLU(inplace=True)
    (7): Dropout(p=0.01, inplace=False)
  )
  </pre>
  <pre>
   OUTPUT BLOCK 
  (maxpool): MaxPool2d(kernel_size=4, stride=2, padding=0, dilation=1, ceil_mode=False)
  (fc): Linear(in_features=512, out_features=10, bias=True)
</pre>




## TRAINING_THE_MODEL
The train function takes the model, device, train_loader, optimizer, and epoch as inputs. It performs the following steps:

* Sets the model to train mode, which enables some layers and operations that are only used during training, such as dropout and batch normalization.
* Creates a progress bar object from the train_loader, which is an iterator that yields batches of data and labels from the training set.
* Initializes two variables to keep track of the number of correct predictions and the number of processed samples.
* Loops over the batches of data and labels, and performs the following steps for each batch:
* Moves the data and labels to the device, which can be either a CPU or a GPU, depending on what is available.
* Calls optimizer.zero_grad() to reset the gradients of the model parameters to zero, because PyTorch accumulates them on subsequent backward passes.
* Passes the data through the model and obtains the predictions (y_pred).
* Calculates the loss between the predictions and the labels using Cross Entropy.
* Appends the loss to the train_losses list for later analysis.
* Performs backpropagation by calling loss.backward(), which computes the gradients of the loss with respect to the model parameters.
* Performs optimization by calling optimizer.step(), which updates the model parameters using the gradients and the chosen optimization algorithm (such as SGD or Adam).
* Updates the progress bar with the current loss, batch index, and accuracy. The accuracy is computed by comparing the predicted class (the index of the max log-probability) with the true class, and summing up the correct predictions and processed samples.
* Appends the accuracy to the train_acc list for later analysis.

The test function takes the model, device, and test_loader as inputs. It performs the following steps:

* Sets the model to eval mode, which disables some layers and operations that are only used during training, such as dropout and batch normalization.
* Initializes two variables to keep track of the total test_loss and the number of correct predictions.
* Uses a torch.no_grad() context manager to disable gradient computation, because we don’t need it during testing and it saves memory and time.
* Loops over the batches of data and labels from the test set, and performs the following steps for each batch:
* Moves the data and labels to the device, which can be either a CPU or a GPU, depending on what is available.
* Passes the data through the model and obtains the output (predictions).
* Adds up the batch loss to the total test loss using the negative log-likelihood loss function (F.nll_loss) with reduction=‘sum’, which means it returns a scalar instead of a vector.
* Compares the predicted class (the index of the max log-probability) with the true class, and sums up the correct predictions.
* Divides the total test loss by the number of samples in the test set to get the average test loss, and appends it to the test_losses list for later analysis.

* creates an instance of the Adam optimizer, which is a popular algorithm that adapts the learning rate for each parameter based on the gradient history and the current gradient. You pass the model parameters, the initial learning rate (lr), and some other hyperparameters to the optimizer constructor. 
* creates an instance of the OneCycleLR scheduler, which is a learning rate policy that cycles the learning rate between two boundaries with a constant frequency. You pass the optimizer, the maximum learning rate (0.01), the number of epochs (30), and the number of steps per epoch (len(train_loader)) to the scheduler constructor.
* Defines a constant for the number of epochs = 30, which is the number of times you iterate over the entire training set.
* Prints out a summary of the average test loss, accuracy, and number of samples in the test set. 

## [LR_SCHEDULAR]
The learning rate finder is a method to discover a good learning rate for most gradient-based optimizers. The LRFinder method can be applied on top of every variant of the stochastic gradient descent, and most types of networks.
* LR FINDER SYNTAX:
<pre>
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

lr_finder = LRFinder(model, optimizer, criterion, device="cuda")
lr_finder.range_test(train_loader, end_lr=10, num_iter=200, step_mode="exp")
lr_finder.plot() # to inspect the loss-learning rate graph

min_loss = min(lr_finder.history["loss"])
max_lr = lr_finder.history["lr"][np.argmin(lr_finder.history["loss"], axis=0)]

print("Min Loss = {}, Max LR = {}".format(min_loss, max_lr))
</pre>
* Always start with maximum learning rates.
* If the Learning Rate is negative either increase the number of epochs or decrease the learning rate. By decreasing the learning rate I am not able to get 90% of accuracy. So I increased the number of epoches in the lr scheduler.
* The following LR scheduler is used in the program
  ![lrfinder](https://github.com/RajidiSahithi/Session_10/blob/main/Images%20S10/LR%20Finder.png)
<pre>
scheduler = OneCycleLR(
        optimizer,
        max_lr=1.74E-03,
        steps_per_epoch=1,
        epochs=26,
        pct_start=5/EPOCHS,
        div_factor=100,
        three_phase=False,
        final_div_factor=100,
        anneal_strategy='linear'
    )
</pre>

## [RESULTS]

I achieved 90% of accuracy in 24 epochs. Initially model is underfitting. As the number of epochs increased it worked good.




#### MODEL SUMMARY
<pre>
cuda
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 64, 32, 32]           1,728
       BatchNorm2d-2           [-1, 64, 32, 32]             128
              ReLU-3           [-1, 64, 32, 32]               0
            Conv2d-4          [-1, 128, 32, 32]          73,728
         MaxPool2d-5          [-1, 128, 16, 16]               0
       BatchNorm2d-6          [-1, 128, 16, 16]             256
              ReLU-7          [-1, 128, 16, 16]               0
            Conv2d-8          [-1, 128, 16, 16]         147,456
       BatchNorm2d-9          [-1, 128, 16, 16]             256
             ReLU-10          [-1, 128, 16, 16]               0
           Conv2d-11          [-1, 128, 16, 16]         147,456
      BatchNorm2d-12          [-1, 128, 16, 16]             256
             ReLU-13          [-1, 128, 16, 16]               0
           Conv2d-14          [-1, 256, 16, 16]         294,912
        MaxPool2d-15            [-1, 256, 8, 8]               0
      BatchNorm2d-16            [-1, 256, 8, 8]             512
             ReLU-17            [-1, 256, 8, 8]               0
           Conv2d-18            [-1, 512, 8, 8]       1,179,648
        MaxPool2d-19            [-1, 512, 4, 4]               0
      BatchNorm2d-20            [-1, 512, 4, 4]           1,024
             ReLU-21            [-1, 512, 4, 4]               0
           Conv2d-22            [-1, 512, 4, 4]       2,359,296
      BatchNorm2d-23            [-1, 512, 4, 4]           1,024
             ReLU-24            [-1, 512, 4, 4]               0
           Conv2d-25            [-1, 512, 4, 4]       2,359,296
      BatchNorm2d-26            [-1, 512, 4, 4]           1,024
             ReLU-27            [-1, 512, 4, 4]               0
        MaxPool2d-28            [-1, 512, 1, 1]               0
           Linear-29                   [-1, 10]           5,130
================================================================
Total params: 6,573,130
Trainable params: 6,573,130
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 6.44
Params size (MB): 25.07
Estimated Total Size (MB): 31.53
----------------------------------------------------------------
</pre>

# Schedular Output:
<pre>
EPOCH: 0
lr=  0.0003723
Loss=1.1109507083892822 Batch_id=97 Accuracy=49.24: 100%|██████████| 98/98 [00:35<00:00,  2.72it/s]

Test set:  Accuracy: 5679/10000 (56.79%)

EPOCH: 1
lr=  0.0007287
Loss=0.8647873401641846 Batch_id=97 Accuracy=64.89: 100%|██████████| 98/98 [00:34<00:00,  2.81it/s]

Test set:  Accuracy: 7312/10000 (73.12%)

EPOCH: 2
lr=  0.0010850999999999999
Loss=0.8662615418434143 Batch_id=97 Accuracy=72.45: 100%|██████████| 98/98 [00:34<00:00,  2.84it/s]

Test set:  Accuracy: 7313/10000 (73.13%)

EPOCH: 3
lr=  0.0014414999999999999
Loss=0.6320058107376099 Batch_id=97 Accuracy=76.11: 100%|██████████| 98/98 [00:34<00:00,  2.84it/s]

Test set:  Accuracy: 7877/10000 (78.77%)

EPOCH: 4
lr=  0.0015449437773279353
Loss=0.5845252871513367 Batch_id=97 Accuracy=79.54: 100%|██████████| 98/98 [00:35<00:00,  2.78it/s]

Test set:  Accuracy: 8028/10000 (80.28%)

EPOCH: 5
lr=  0.0014677045384615386
Loss=0.5801482200622559 Batch_id=97 Accuracy=82.27: 100%|██████████| 98/98 [00:35<00:00,  2.80it/s]

Test set:  Accuracy: 8078/10000 (80.78%)

EPOCH: 6
lr=  0.0013904652995951417
Loss=0.45305952429771423 Batch_id=97 Accuracy=83.84: 100%|██████████| 98/98 [00:34<00:00,  2.80it/s]

Test set:  Accuracy: 8616/10000 (86.16%)

EPOCH: 7
lr=  0.001313226060728745
Loss=0.36993470788002014 Batch_id=97 Accuracy=85.62: 100%|██████████| 98/98 [00:34<00:00,  2.85it/s]

Test set:  Accuracy: 8342/10000 (83.42%)

EPOCH: 8
lr=  0.0012359868218623483
Loss=0.36582207679748535 Batch_id=97 Accuracy=86.78: 100%|██████████| 98/98 [00:34<00:00,  2.80it/s]

Test set:  Accuracy: 8730/10000 (87.30%)

EPOCH: 9
lr=  0.0011587475829959513
Loss=0.3040536046028137 Batch_id=97 Accuracy=88.13: 100%|██████████| 98/98 [00:34<00:00,  2.81it/s]

Test set:  Accuracy: 8489/10000 (84.89%)

EPOCH: 10
lr=  0.0010815083441295546
Loss=0.35869351029396057 Batch_id=97 Accuracy=88.50: 100%|██████████| 98/98 [00:34<00:00,  2.86it/s]

Test set:  Accuracy: 8822/10000 (88.22%)

EPOCH: 11
lr=  0.001004269105263158
Loss=0.3063914179801941 Batch_id=97 Accuracy=89.52: 100%|██████████| 98/98 [00:35<00:00,  2.77it/s]

Test set:  Accuracy: 8797/10000 (87.97%)

EPOCH: 12
lr=  0.0009270298663967612
Loss=0.2671511173248291 Batch_id=97 Accuracy=90.07: 100%|██████████| 98/98 [00:34<00:00,  2.82it/s]

Test set:  Accuracy: 8858/10000 (88.58%)

EPOCH: 13
lr=  0.0008497906275303645
Loss=0.3242798149585724 Batch_id=97 Accuracy=91.12: 100%|██████████| 98/98 [00:34<00:00,  2.85it/s]

Test set:  Accuracy: 8772/10000 (87.72%)

EPOCH: 14
lr=  0.0007725513886639677
Loss=0.25662508606910706 Batch_id=97 Accuracy=91.72: 100%|██████████| 98/98 [00:33<00:00,  2.89it/s]

Test set:  Accuracy: 8875/10000 (88.75%)

EPOCH: 15
lr=  0.000695312149797571
Loss=0.25481343269348145 Batch_id=97 Accuracy=92.14: 100%|██████████| 98/98 [00:33<00:00,  2.90it/s]

Test set:  Accuracy: 8739/10000 (87.39%)

EPOCH: 16
lr=  0.0006180729109311741
Loss=0.18919914960861206 Batch_id=97 Accuracy=92.56: 100%|██████████| 98/98 [00:35<00:00,  2.79it/s]

Test set:  Accuracy: 8978/10000 (89.78%)

EPOCH: 17
lr=  0.0005408336720647774
Loss=0.2174367904663086 Batch_id=97 Accuracy=93.10: 100%|██████████| 98/98 [00:34<00:00,  2.84it/s]

Test set:  Accuracy: 8932/10000 (89.32%)

EPOCH: 18
lr=  0.0004635944331983805
Loss=0.15855424106121063 Batch_id=97 Accuracy=93.99: 100%|██████████| 98/98 [00:34<00:00,  2.83it/s]

Test set:  Accuracy: 9063/10000 (90.63%)

EPOCH: 19
lr=  0.0003863551943319838
Loss=0.150477334856987 Batch_id=97 Accuracy=94.38: 100%|██████████| 98/98 [00:34<00:00,  2.83it/s]

Test set:  Accuracy: 9041/10000 (90.41%)

EPOCH: 20
lr=  0.00030911595546558707
Loss=0.14337743818759918 Batch_id=97 Accuracy=95.03: 100%|██████████| 98/98 [00:34<00:00,  2.83it/s]

Test set:  Accuracy: 9120/10000 (91.20%)

EPOCH: 21
lr=  0.00023187671659919036
Loss=0.16122034192085266 Batch_id=97 Accuracy=95.62: 100%|██████████| 98/98 [00:34<00:00,  2.82it/s]

Test set:  Accuracy: 9205/10000 (92.05%)

EPOCH: 22
lr=  0.00015463747773279365
Loss=0.07756074517965317 Batch_id=97 Accuracy=96.16: 100%|██████████| 98/98 [00:34<00:00,  2.86it/s]

Test set:  Accuracy: 9229/10000 (92.29%)

EPOCH: 23
lr=  7.739823886639672e-05
Loss=0.14669601619243622 Batch_id=97 Accuracy=96.50: 100%|██████████| 98/98 [00:33<00:00,  2.90it/s]

Test set:  Accuracy: 9267/10000 (92.67%)


</pre>
![output](https://github.com/RajidiSahithi/Session_10/blob/main/Images%20S10/output.png)
# ANALYSIS:
* Accuracy > 90% that is 91.12%
* RF reached upto 90
* LRFinder is used to improve the accuracy and lr is varying for  every batch
* Max Learning Rate = 0.012173827277396614 (from LR Plot) and = 0.0016906931902834009 from the schedular output
* Suggested Learning Rate = 1.74E-03
* Minimum Learning Rate = 1.74E-05 from finder and = 8.469995951417009e-05 from the schedular output
  ![rf](https://github.com/RajidiSahithi/Session_10/blob/main/Images%20S10/RF%20Calculation.png)



