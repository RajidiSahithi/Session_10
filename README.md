# DESIGN A CNN USING DILATION CONVOLUTION AND DEPTH WISE CONVOLUTION

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
* Use ADAM, and CrossEntropyLoss
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
* NumPy is used for for numerical operations.The torch library is used to import Pytorch.
* Pytorch has an nn component that is used for the abstraction of machine learning operations. 
* The torchvision library is used so that to import the CIFAR-10 dataset. This library has many image datasets and is widely used for research. The transforms can be imported to resize the image to equal size for all the images. 
* The optim is used train the neural Networks.
* MATLAB libraries are imported to plot the graphs and arrange the figures with labelling
* Albumenations are imported for Middle Man's Data Augmentation Strategy
* cv2 is imported 
* torch_lr_finder is imported for finding the maximum and minimum learning rate

## SET_THE_ALBUMENTATIONS
* cv2.setNumThreads(0) sets the number of threads used by OpenCV to 0. This is done to avoid a deadlock when using OpenCV’s resize method with PyTorch’s dataloader1.

* cv2.ocl.setUseOpenCL(False) disables the usage of OpenCL in OpenCV2 and is used when you want to disable the usage of OpenCL.

* The  class is inherited from torchvision.datasets.CIFAR10. It overrides the __init__ and __getitem__ methods of the parent class. The __getitem__ method returns an image and its label after applying a transformation to the image3. (This is to be done while using Albumenations)


## DATA_AUGMENTATIONS
For this import albumentations as A

Middle-Class Man's Data Augmentation Strategy is used. Like
### Normalize
<pre>
Syntax:
     A.Normalize(
            mean = (0.4914, 0.4822, 0.4465),
            std = (0.2470, 0.2435, 0.2616), always_apply = True)

Normalization is a common technique used in deep learning to scale the pixel values of an image to a standard range. This is done to ensure that the input features have similar ranges and are centered around zero. 
Normalization is done with respect to mean and standard Deviation.
For CIFAR10 (RGB) will have 3 means and 3 standard devivation that is equal to 
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
Randomly apply affine transforms: translate, scale and rotate the input.

    Args:
        shift_limit ((float, float) or float): shift factor range for both height and width. If shift_limit
            is a single float value, the range will be (-shift_limit, shift_limit). Absolute values for lower and
            upper bounds should lie in range [0, 1]. Default: (-0.0625, 0.0625).
        scale_limit ((float, float) or float): scaling factor range. If scale_limit is a single float value, the
            range will be (-scale_limit, scale_limit). Note that the scale_limit will be biased by 1.
            If scale_limit is a tuple, like (low, high), sampling will be done from the range (1 + low, 1 + high).
            Default: (-0.1, 0.1).
        rotate_limit ((int, int) or int): rotation range. If rotate_limit is a single int value, the
            range will be (-rotate_limit, rotate_limit). Default: (-45, 45).
        p(float): probability of applying the transform. Default: 0.5.

Applied only to Training data
</pre>
### PadIfNeeded
<pre>
Syntax:
    A.PadIfNeeded(min_height=36, min_width=36, p=1.0),
PadIfNeeded is an image augmentation technique that pads the input image on all four sides if the side is less than the desired number. The desired number is specified by the min_height and min_width parameters. In this case padding is equal to 4.

</pre>
### RandomCrop
<pre>
Syntax:
  A.RandomCrop(height=32, width=32, always_apply = False,p=1.0),

RandomCrop is an image augmentation technique that crops a random part of the input and rescales it to some size without loss of bounding boxes. The height and width parameters specify the size of the crop. In this case iamge is cropped to size 32 X 32
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
 It is similar to cutout

    Args:
        max_holes(int): The maximum number of rectangular regions to be masked. (for CIFAR10 Dataset its 32X32)
        max_height(int): The maximum height of the rectangular regions. 
        max_width(int): The maximum width of the rectangular regions.
        min_holes(int): The minimum number of rectangular regions to be masked.
        min_height(int): The minimum height of the rectangular regions.
        min_width(int): The minimum width of the rectangular regions.
        fill_value(float): The value to be filled in the masked region. It can be a tuple or a single value . 
            It is usually equal to the mean of dataset for CIFAR10 its (0.4914, 0.4822, 0.4465)
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
![alt text]sample images train loader

## CNN_MODEL

#### MODEL
<pre>
Preparation Layer:
 This Block Contains 1 Convolution layers with Kernal Size 3X3 and Stride is 1 . (each layer has Batch Normalization, Activation Function (Relu) and Dropout)
   (conv1): Sequential(
    (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(inplace=True)
  )
</pre>
<pre>
Layer 1:
  This Block Contains 4 Convolution layers with Kernal Size 3X3 and Stride is 1 all layers and maxpooling with stride 2 and kernal size = 2
  
  
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
The learning rate finder is a method to discover a good learning rate for most gradient based optimizers. The LRFinder method can be applied on top of every variant of the stochastic gradient descent, and most types of networks.
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
* Always starting with maximum learning rates.
* If Learning Rate is negative either increase the number of epochs or decrease learning rate. By decreasing the learning rate Iam not able to get 90% of accuracy. So I incresed the number of epoches in lr schedular.
* The following LR Schedular is used the program
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

I achieved 90% of accuracy in 24 epochs. Initially model is under fitting. As number of epochs increased it worked good.




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
lr=  0.0004074226415094339
Loss=1.0863311290740967 Batch_id=97 Accuracy=49.47: 100%|██████████| 98/98 [00:35<00:00,  2.79it/s]

Test set:  Accuracy: 5929/10000 (59.29%)

EPOCH: 1
lr=  0.0007974452830188678
Loss=0.8722756505012512 Batch_id=97 Accuracy=65.48: 100%|██████████| 98/98 [00:34<00:00,  2.84it/s]

Test set:  Accuracy: 7199/10000 (71.99%)

EPOCH: 2
lr=  0.0011874679245283018
Loss=0.8201348185539246 Batch_id=97 Accuracy=72.48: 100%|██████████| 98/98 [00:34<00:00,  2.86it/s]

Test set:  Accuracy: 7686/10000 (76.86%)

EPOCH: 3
lr=  0.0015774905660377357
Loss=0.671578586101532 Batch_id=97 Accuracy=76.49: 100%|██████████| 98/98 [00:35<00:00,  2.80it/s]

Test set:  Accuracy: 7849/10000 (78.49%)

EPOCH: 4
lr=  0.0016906931902834009
Loss=0.6401821970939636 Batch_id=97 Accuracy=80.13: 100%|██████████| 98/98 [00:34<00:00,  2.81it/s]

Test set:  Accuracy: 7519/10000 (75.19%)

EPOCH: 5
lr=  0.001606167230769231
Loss=0.5577336549758911 Batch_id=97 Accuracy=82.30: 100%|██████████| 98/98 [00:34<00:00,  2.86it/s]

Test set:  Accuracy: 8097/10000 (80.97%)

EPOCH: 6
lr=  0.0015216412712550607
Loss=0.44225797057151794 Batch_id=97 Accuracy=84.09: 100%|██████████| 98/98 [00:34<00:00,  2.87it/s]

Test set:  Accuracy: 8543/10000 (85.43%)

EPOCH: 7
lr=  0.0014371153117408906
Loss=0.39895716309547424 Batch_id=97 Accuracy=85.63: 100%|██████████| 98/98 [00:35<00:00,  2.80it/s]

Test set:  Accuracy: 8371/10000 (83.71%)

EPOCH: 8
lr=  0.0013525893522267206
Loss=0.3777617812156677 Batch_id=97 Accuracy=86.80: 100%|██████████| 98/98 [00:35<00:00,  2.76it/s]

Test set:  Accuracy: 8669/10000 (86.69%)

EPOCH: 9
lr=  0.0012680633927125507
Loss=0.41653934121131897 Batch_id=97 Accuracy=87.61: 100%|██████████| 98/98 [00:34<00:00,  2.84it/s]

Test set:  Accuracy: 8710/10000 (87.10%)

EPOCH: 10
lr=  0.0011835374331983805
Loss=0.33829236030578613 Batch_id=97 Accuracy=88.72: 100%|██████████| 98/98 [00:34<00:00,  2.82it/s]

Test set:  Accuracy: 8730/10000 (87.30%)

EPOCH: 11
lr=  0.0010990114736842106
Loss=0.32964614033699036 Batch_id=97 Accuracy=89.20: 100%|██████████| 98/98 [00:35<00:00,  2.78it/s]

Test set:  Accuracy: 8911/10000 (89.11%)

EPOCH: 12
lr=  0.0010144855141700406
Loss=0.2808605134487152 Batch_id=97 Accuracy=90.09: 100%|██████████| 98/98 [00:34<00:00,  2.83it/s]

Test set:  Accuracy: 8826/10000 (88.26%)

EPOCH: 13
lr=  0.0009299595546558704
Loss=0.25487521290779114 Batch_id=97 Accuracy=90.50: 100%|██████████| 98/98 [00:34<00:00,  2.83it/s]

Test set:  Accuracy: 8925/10000 (89.25%)

EPOCH: 14
lr=  0.0008454335951417004
Loss=0.23481881618499756 Batch_id=97 Accuracy=91.53: 100%|██████████| 98/98 [00:34<00:00,  2.83it/s]

Test set:  Accuracy: 8855/10000 (88.55%)

EPOCH: 15
lr=  0.0007609076356275303
Loss=0.22415052354335785 Batch_id=97 Accuracy=92.07: 100%|██████████| 98/98 [00:35<00:00,  2.79it/s]

Test set:  Accuracy: 8981/10000 (89.81%)

EPOCH: 16
lr=  0.0006763816761133604
Loss=0.2578561007976532 Batch_id=97 Accuracy=92.80: 100%|██████████| 98/98 [00:35<00:00,  2.74it/s]

Test set:  Accuracy: 9126/10000 (91.26%)

EPOCH: 17
lr=  0.0005918557165991902
Loss=0.17052705585956573 Batch_id=97 Accuracy=93.30: 100%|██████████| 98/98 [00:34<00:00,  2.83it/s]

Test set:  Accuracy: 9112/10000 (91.12%)

EPOCH: 18
lr=  0.00050732975708502
Loss=0.21183858811855316 Batch_id=97 Accuracy=93.99: 100%|██████████| 98/98 [00:34<00:00,  2.86it/s]

Test set:  Accuracy: 9086/10000 (90.86%)

EPOCH: 19
lr=  0.0004228037975708501
Loss=0.1778084933757782 Batch_id=97 Accuracy=94.41: 100%|██████████| 98/98 [00:35<00:00,  2.79it/s]

Test set:  Accuracy: 9169/10000 (91.69%)

EPOCH: 20
lr=  0.00033827783805668015
Loss=0.18209674954414368 Batch_id=97 Accuracy=95.05: 100%|██████████| 98/98 [00:34<00:00,  2.80it/s]

Test set:  Accuracy: 9100/10000 (91.00%)

EPOCH: 21
lr=  0.0002537518785425102
Loss=0.13834111392498016 Batch_id=97 Accuracy=95.69: 100%|██████████| 98/98 [00:34<00:00,  2.82it/s]

Test set:  Accuracy: 9250/10000 (92.50%)

EPOCH: 22
lr=  0.00016922591902834004
Loss=0.12079329788684845 Batch_id=97 Accuracy=96.03: 100%|██████████| 98/98 [00:34<00:00,  2.85it/s]

Test set:  Accuracy: 9250/10000 (92.50%)

EPOCH: 23
lr=  8.469995951417009e-05
Loss=0.12226494401693344 Batch_id=97 Accuracy=96.45: 100%|██████████| 98/98 [00:34<00:00,  2.81it/s]

Test set:  Accuracy: 9264/10000 (92.64%)

</pre>
# ANALYSIS:
* Accuracy > 90% that is 91.12%
* RF reached upto 90
* LRFinder is used to improve the accuracy and lr is varying for  every batch
* Max Learning Rate = 0.012173827277396614 (from LR Plot) and = 0.0016906931902834009 from the schedular output
* Suggested Learning Rate = 1.74E-03
* Mininum Learning Rate = 1.74E-05 from finder and = 8.469995951417009e-05 from the schedular output



