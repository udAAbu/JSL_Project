# JSL_Project
Predicting Joint space loss(JSL) progression with deep learning

### Preliminary work
* Convert all the DICOM images to PNG, and split in half to extract the left knee and right knee. 
* Feed the single knee images to YOLO-V4 to automatically plot the bounding box on the knee joint and crop the bounding region.

### Illustration of the three notebooks:
* All three notebooks follow this workflow
  1. data preprocessing
  2. loading data in batch using custom Dataset and Dataloader
  3. define model architecture
  4. create training and evaluation function
  5. initialize model, begin training and evaluation
  6. visualize model using saliency map
  7. save model

* [Pretraining_on_KL_grade](https://github.com/udAAbu/JSL_Project/blob/main/Pretraining_on_KL_grade.ipynb): Before training the CNN model to predict **joint space loss (JSL)** progression, I first train the model to predict 5-level **KL_grade** (~0.72 accuracy). There are several reasons to this:
  * KL_grade is one of the most important factors in determining the Knee Osteoarthritis severity. 
  * By first training model to predict KL_grade, the model can learn fruitful features that can help our downstream task of predicting the JSL progression. 
  * We have more labels on KL_grade (8000+) than on JSL progression(4000+), which helps the large network train and stablize.
  * Give us more efficient training in later work and the training convergence will be faster.

* [radio_joint](https://github.com/udAAbu/JSL_Project/blob/main/radio_joint.ipynb): This notebook follows the previous one. After pre-trained our model on KL_grade prediction, I transfered this model to predict the JSL progression. There are two main parts in this notebook:
  * **Part 1:** This part only use raw images to train the CNN without using any risk factors. You can decide whether to use the model pretrained on KL_grade to start training. Empirically, using the pre-trained model as a starting point gives us more efficient training and better performance (~0.7 AUC). Once the model has been fully trained, you can begin to work on part 2.
  * **Part 2:** This part uses raw images together with risk factors to train the model. You can decide whether to freeze the CNN weights or unfreeze them. Empirically, freezing the CNN weights and only train the fully-connected layers after the concatnation typically works better (~0.72 AUC). 

* [Multi-Tasking](https://github.com/udAAbu/JSL_Project/blob/main/Multi_tasking.ipynb): This notebook is seperate from the previous two, I took a different approach to see whether the performance can be improved. I trained a CNN to predict the KL_grade and JSL progression at the same time. I attached two head layers to the end of the CNN. One has 5 output nodes to predict KL_grade, and the other one has 1 output nodes to predict JSL progression. However, the result does not show any improvement (~0.7 AUC).

### Thing I have tried to improve performace:
* batch size (8, 16, 32, 64) (higher batch size will cause GPU memory issue)
* all kinds of learning rate, optimizer, scheduler
* model type and architecture (e.g. Densenet, VGG, Resnet, etc) (All pretrained on ImageNet)
* train on both balanced dataset and unbalanced dataset
* further cropping the image
* data augmentation
* two output nodes with softmax and one output nodes with sigmoid.
* use ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) to normalize image tensors.
* some other trivial details



