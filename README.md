# Risk assessment of JSL progression
Predicting joint space loss (JSL) progression with deep learning

## Introduction
* Knee osteoarthritis is a condition where the cartilage in the knee degenerates, or break down, which will results in pain, swelling, stiffness and decreased ability to move. This project is focused on developing a risk assessment deep learning model that can automatically evaluate the risk of progression of joint space loss using the knee's baseline X-rays image together with the patient's clinical risk factors including age, gender, race, BMI, KL-grade and etc. The model can help to identify at the early stage those individuals who are at a high risk of developing the knee OA, and provide them with more appropriate and effective treatments to prevent the progression of joint space loss. Note that the models are not entirely desireale and optimal yet, and we are still working to improve the performance.

## Dataset
* The dataset was provided by Osteoarthritis Initiative (OAI) database, a multi-center study which collected longitudinal clinical and imaging data over a 9-year follow-up
period in 4796 subjects between the ages of 45 and 75 years with or at high risk for knee OA. Among the all the 9592 knees from 4796 subjects, 4163 knees has the baseline X-ray images and the following baseline clinical risk factors available:
  * **Age, Gender, Race, BMI, Knee injury history** at baseline.
  * **KL-grade** of knee OA provided by central reading at baseline.
  * **anatomic axis alignment measurements** provided by central reading at baseline.
  * **minimum medial joint space width measurements** provided by central reading at baseline and 48-month follow-up. (This will be used for defining joint space loss and labeling our target variable)
 
* Definitive progression of **radiographic joint space loss (JSL)** was defined according to the National Institute of Health OA Biomarkers Consortium Project as a greater than or equal to 0.7 mm decrease in minimum medial joint space width measurements obtained between baseline and 48-month follow-up. As a result, we have **2812 knees with no-progression and 1351 knees with progression**. 

* Here is an image example illustrating **radiographic joint space loss(JSL)**: 
  * <img src="https://www.shawephysio.com/wp-content/uploads/2018/05/knee-arthritis.png" width="300" height="300">

## Preliminary work and image preprocessing
1. Original images are in DICOM format containing both knees for one patient, and we convert them into PNG format and split them in half to extract the left knee and right knee.
2. We adapted [YOLO-V4](https://github.com/udAAbu/yolo-v4) (You Only Look Once) to train a fully-automated joint cropping CNN, which help us automatically identify the joint of a knee and crop it. Our custom YOLO model can be found in this repo: https://github.com/udAAbu/yolo-v4.
3. These cropped images will become the final inputs to our CNN model.
<img style="text-align:center;" src="https://github.com/udAAbu/JSL_Project/blob/main/github_images/Image.png">

***The diagram above shows how we split the image, feed the single knee images to YOLO-V4 to automatically plot the bounding box around the knee joint and crop the bounding region***

## Methodology:
- **Kellgren-Lawrence grade (KL-grade)**:
  - KL-grade is a scale that measures the currect stage of OA. It's determined by the central reading from the baseline images. There are 5 grades in total (0-4), and the severity of OA increases as the number goes up. The following images provides a illustration: 
  - <img src="https://www.researchgate.net/profile/Joseph_Antony3/publication/315695691/figure/fig6/AS:668531486101508@1536401735381/The-KL-grading-system-to-assess-the-severity-of-knee-OA.ppm" width="600" height="350">. 
  - In our two methods mentioned below, we will adopt the KL-grade in 2 different ways, and we hope that with the help of KL-grade, the model can learn more fruitful and informative features that can help our downstream task of predicting the JSL progression. 

- **Method 1**:
  - My first approach is to formulate the problem in a multi-tasking setting to predict the KL-grade and the JSL progression at the same time. We first use CNN as a feature extractor, and two different fully-connected networks are attached to the end of the CNN to make the classification. One has 5 output nodes with a softmax activation responsible for KL_grade classification, and the other one has 1 output nodes with a sigmoid activation to predict JSL progression. The following image shows the architecture of the whole network. 
 - 
 - 
Before getting into predicting the **joint space loss (JSL)** progression directly, we first trained the model to predict 5-level **KL_grade** (~0.72 accuracy). There are several reasons to this:
  * KL_grade is one of the most important factors in determining the Knee Osteoarthritis severity, and it's labeled by central reading from the baseline images. Here is a image shows the image corresponding to KL-grade 1-4: <img src="https://www.researchgate.net/profile/Joseph_Antony3/publication/315695691/figure/fig6/AS:668531486101508@1536401735381/The-KL-grading-system-to-assess-the-severity-of-knee-OA.ppm">
  * We hope that by first training model to predict KL_grade, the model can learn fruitful features that can help our downstream task on predicting the JSL progression. 
  * We have more labels on KL_grade (8000+) than on JSL progression(4000+), which helps the large network train and stablize, and give us more efficient training 
  * Give us more efficient training in later work and the training convergence will be faster.

<img src="https://www.researchgate.net/profile/Joseph_Antony3/publication/315695691/figure/fig6/AS:668531486101508@1536401735381/The-KL-grading-system-to-assess-the-severity-of-knee-OA.ppm">

### Several notes on running the notebooks on Google Colab
* You need to change the all paths appeared in the notebooks to the paths in your google drive.
* If you use the KL_grade pre-trained model, [Pretraining_on_KL_grade](https://github.com/udAAbu/JSL_Project/blob/main/Pretraining_on_KL_grade.ipynb) should be executed before running [radio_joint](https://github.com/udAAbu/JSL_Project/blob/main/radio_joint.ipynb).

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

### Thing have been tried to improve performance:
* batch size (8, 16, 32, 64) (higher batch size will cause GPU memory issue)
* all kinds of learning rate, optimizer, scheduler
* model type and architecture (e.g. Densenet, VGG, Resnet, etc) (All pretrained on ImageNet)
* train on both balanced dataset and unbalanced dataset
* further cropping the image
* data augmentation
* two output nodes with softmax and one output nodes with sigmoid.
* use ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) to normalize image tensors.
* some other trivial details



