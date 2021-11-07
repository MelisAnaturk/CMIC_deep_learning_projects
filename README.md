# Kick starting your deep learning project on CMIC's HPC cluster

## 1. Introduction
This page provides some tips on getting started with projects that aim to apply deep learning to MRI data. It also includes specific instructions on how to set things up on UCL Centre for Medical Image Computing (CMIC) high-performance computing (HPC) cluster (i.e. ```comic```) for new starters. Please note that there is a bit of overlap between this page and UCL DRC Neuroimaging Analysis Wiki page on [deep learning](https://wiki.ucl.ac.uk/pages/viewpage.action?pageId=181248279) (written by myself & Sophie Martin).

***Disclaimer: The information available on this page is by no means exhaustive and may eventually become outdated. If you do find any mistakes or inaccuracies please do get in touch with me at melis.anaturk.14@ucl.ac.uk or create a pull request.***

## 2. General overview of deep learning
Deep learning is a subfield of Machine learning (ML) where algorithms consisting of a series of layers (i.e. an input layer, hidden layers and output layer, otherwise referred to the algorithms "architecture") are trained to complete specific tasks, such as making predictions or classifications about data. While classical ML (e.g., random forests, elastic net) require features to already be extracted from MRI images (e.g., FreeSurfer measures of cortical thickness, surface area and cortical/subcortical volumes), deep learning algorthims are able to **learn the features** directly from raw images, without the need for preprocessing the data beforehand. Deep learning algorthims often (but not necessarily always!) outperform classifical ML methods in a range of tasks.

### 2.1 Some example projects where deep learning models have been applied:
> 1. Automated brain segmentation (e.g., SynthSeg)
> 2. Building predictive models of biological 'brain age' (e.g., SFCN based on T1-weighted images)
> 3. Classifying individuals into disease categories (e.g., healthy, MCI, dementia)

### 2.2 Resources 
An in-depth introduction to deep learning is beyond the scope of this page as there are already several excellent resources available on topic:
> 1. [Brief introductory article on deep learning](https://machinelearningmastery.com/what-is-deep-learning/)
> 2. [Neural Networks and Deep learning course run by Andrew Ng on Coursera](https://www.coursera.org/specializations/deep-learning?utm_source=gg&utm_medium=sem&utm_campaign=17-DeepLearning-ROW&utm_content=17-DeepLearning-ROW&campaignid=6465471773&adgroupid=77656689495&device=c&keyword=online%20deep%20learning%20classes&matchtype=b&network=g&devicemodel=&adpostion=&creativeid=506750650449&hide_mobile_promo&gclid=Cj0KCQjw8p2MBhCiARIsADDUFVEMeZx6yWRlU9yi0BUlTKpULy8GdWxtVtbJB62kIOIpwm5CAfLQzcsaAtU7EALw_wcB)
> 3. [Deep learning (Goodfellow, et al. 2016)](https://www.deeplearningbook.org/)
> 4. An overview of deep learning models and their applications to MRI images (cite)

You may also be able to request to audit the following courses (*UCL staff and students only*):
> 1. MPHY0025: Information Processing in Medical Imaging (contact: James Cole, james.cole@ucl.ac.uk)
> 2. MPHY0041: Machine Learning in Medical Imaging (contact: Andre Altmann, a.altmann@ucl.ac.uk)
> 3. COMP0090: Introduction to Deep Learning (contact: Yipeng Hu, yipeng.hu@ucl.ac.uk)

Useful packages to be aware of:
1. MONAI: a PyTorch-based, open-source framework for deep learning in healthcare imaging:
>   1.1 Series of videos from a 2-day bootcamp introduction to MONAI (2020)   
>   1.2 Various tutorials on 2D and 3D classification and segmentation examples   
    
2. Deep Learning Tool Kit (DLTK): an open-source deep learning library for medical imaging:
>    2.1 Introduction to Tensorflow and biomedical imaging analysis    
>    2.2 Several tutorials covering the basics of reading in images, data augmentation and building a model
    
3. Torch I/0: image processing and data augmentation

## 3. Getting started on the cluster
You can find advice on working with the HPC cluster (i.e., ```comic```) here and https://github.com/UCL/ECON-CLUSTER. This includes links for things like:

> 1. Setting up an account 
> 2. Familiarise yourself with the Sun Grid Engine (SGE)
> 3. Understanding cluster use etiquette
> 4. Data storage

**Note: To view internal webpages you must have a CS account or request the username and password by emailing cluster-accounts@cs.ucl.ac.uk.**

4. Install a copy of all of the python packages needed for your deep learning project using ```pip``` or ```pip3```.  First, prepare a text file that contains a list of all required packages (in the following format: ```<package_name>==<version>```) and save this to your scratch (e.g., “requirements.txt”):
```
...
Markdown==3.3.3
MarkupSafe==1.1.1
matplotlib==3.2.2
matplotlib-venn==0.11.6
missingno==0.4.2
mistune==0.8.4
mizani==0.6.0
mkl==2019.0
mlxtend==0.14.0
monai==0.4.0
more-itertools==8.7.0
moviepy==0.2.3.5
mpmath==1.1.0
msgpack==1.0.2
...
```

Once you have an exhaustive list of packages, run the following in your command line:

```python -m pip install -r requirements.txt --user your_username```

**Note: It’s worth checking whether the packages required are already installed in /share/apps/python-3.8.5-shared/lib before doing this step.**

## 4. Downloading and organising your data 
### 4.1 Importing/downloading data
Getting data onto the cluster is a straight forward process. For example you can use ```wget``` for publicly available dataset e.g. for IXI data:    
``` 
wget -cq http://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI/IXI-T1.tar 
wget http://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI/IXI.xls
```

For researchers with approved UK Biobank projects: specific instructions on downloading 'bulk data' using helper programs (e.g. ```ukbfetch```) are available at this link.

You can also import a copy onto the cluster using ```scp``` (example provided below). 
 
### 4.2 Organisation of data directory
It is important to spend some time thinking about the overall structure of your data directory once you've downloaded or imported it onto the cluster. 

For example, if predicting whether an individual belongs to a specific category (e.g., female/male, patient/control) you could organise it as follows:
 
```
.    
├── labels.csv   
├── Females
│    ├── Sub_101_T1.nii.gz
│    ├── Sub_105_T1.nii.gz
│    ...
│    └── Sub_N_T1.nii.gz
└── Males
     ├── Sub_101_T1.nii.gz
     ...
     └── Sub_103_T1.nii.gz
```
Alternatively, if you are predicting a continuous variable (e.g., age), then you could organise your data directory follows:
  
```
.    
├── labels.csv  
├── Sub_101
│   └── T1.nii.gz
├── Sub_102
│   └── T1.nii.gz
├── Sub_103
│   └── T1.nii.gz
├── Sub_104
│   └── T1.nii.gz
``` 
    
Where ```labels.csv``` contains the ID, label and file pathway for each participant in your sample, which will be necessary for when you are training and evaluating your model.
    
## 5. Prepare your script
Put together your ```python``` script or ```Jupyter``` notebook. Some important considerations here include the model architecture given the task at hand and the type of data augmentation to apply during training, which can help reduce the risk of overfitting to your training set. I've previously used Google Colab as it allows free (albeit limited) access to a GPU node and debug your script. For some example scripts for Segmentation and Classification using the MONAI framework here. 

If you are using Google Colab or an equivalent and want to test your code, it's advisable to use publicly available data. Some examples include IXI dataset or OASIS.

## 6. Move your script to cluster
Once you’re happy that things generally work – the next step is to import your script to the cluster. 
There are several ways to do this (e.g., ```rsync```), but I tend to use ```scp``` for moving my files between my laptop and the cluster:    
1. Type the following into a new terminal (replace username with your details):
```ssh -L 2222:comic.cs.ucl.ac.uk:22 username@tails.cs.ucl.ac.uk```
  
2. Then type the following into another terminal (logged into the cluster):
```scp -P 2222 /Users/ExampleName/Documents/example.py   manaturk@localhost://home/username/scripts```

## 7. Submit bash script to SGE scheduler or request an interactive session
```example.sh```  contains an example script that you can use to submit your DL job

 If you need a short interactive session for debugging you can request using ```qrsh```:
 
``` qrsh -l tmem=4G,gpu=true,h_rt=0:30:0 -pe gpu 1``` ### CHECK THIS

This command will log you into a GPU node for 30 minutes and allow you to use up to 4G of memory.
   
## 8. Evaluating model performance
You can monitor the progress of training through learning curves of your model. Several ways to do this, including using the interactive dashboard created by ```TensorFlow``` or creating simple plots of a performance metric (e.g., Mean Absolute Error) over training epochs using ```matplotlib```.

Unfortunately, there isn’t a way to view your plots in the cluster (that I’m aware of), so you will need to copy this onto your local desktop. To do this:
1. Type the following into a new command line terminal (replace username with your details):
  
  ```ssh -L 2222:comic.cs.ucl.ac.uk:22 username@tails.cs.ucl.ac.uk```
  
2. In another terminal (make sure you are not logged into the cluster):
  
```scp -P 2222 username@localhost:/home/username/plot.pdf /Users/ExampleName/deep_learning_project/```
