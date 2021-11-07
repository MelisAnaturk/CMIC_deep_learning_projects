# A guide to kick starting your deep learning project on UCL CMIC's HPC cluster

## 1. Introduction
This page provides some tips on getting started with projects that aim to apply deep learning models based on 3D medical images acquired using Magnetic Resonance Imaging (MRI). It also includes specific instructions on how to set things up on UCL CMIC HPC cluster for new starters. Note that there is quite an overlap between this page and UCL DRC Neuroimaging Analysis Wiki page on [deep learning](https://wiki.ucl.ac.uk/pages/viewpage.action?pageId=181248279) (written by myself & Sophie Martin).

***Disclaimer: This is by no means an exhaustive introduction and the information available on this page may become outdated. If you do find any mistakes or inaccuracies please do get in touch with me at melis.anaturk.14@ucl.ac.uk or create a pull request.***

## 2. General overview of deep learning
Deep learning is a subfield of Machine learning where algorithms are structured into layers of nodes or "neurons", otherwise refer "neural networks" (consisting of an input layer, hidden layers and output layer, see example below). While classical machine learning requires features to already be extracted from the images (e.g., FreeSurfer measures of cortical thickness, surface area and cortical/subcortical volumes), deep learning models are able to **learn the features** directly to the raw images, without human intervention. Deep learning models have led to some of the most accurate predictions..

### 2.1 Some example projects where deep learning models have been applied:
1. Automated brain segmentation (e.g., SynthSeg)
2. Building predictive models of biological 'brain age' (e.g., SFCN based on T1-weighted images)
3. Classifying individuals into disease categories (e.g., healthy, MCI, dementia)

### 2.2 Resources 
An in-depth introduction to deep learning is beyond the scope of this page as there are already several excellent resources that are publicly available on topic:
1. Brief introductory article on deep learning
2. Neural Networks and Deep learning course run by Andrew Ng on Coursera
3. Deep learning (Goodfellow, et al. 2016)

You may also be able to request to audit the following courses (*UCL staff and students only*):
1. MPHY0025: Information Processing in Medical Imaging (contact: James Cole, james.cole@ucl.ac.uk)
2. MPHY0041: Machine Learning in Medical Imaging (contact: Andre Altmann, a.altmann@ucl.ac.uk)
3. COMP0090: Introduction to Deep Learning (contact: Yipeng Hu, yipeng.hu@ucl.ac.uk)

Useful packages to be aware of:
1. MONAI: a PyTorch-based, open-source framework for deep learning in healthcare imaging: 

    1.1 Series of videos from a 2-day bootcamp introduction to MONAI (2020)
    
    3.2 Various tutorials on 2D and 3D classification and segmentation examples
    
2. Deep Learning Tool Kit (DLTK): an open-source deep learning library for medical imaging:

    2.1 Introduction to Tensorflow and biomedical imaging analysis
    
    4.2 Several tutorials covering the basics of reading in images, data augmentation and building a model

## 3. Getting started
You can find advice on working with the Computer Science high-performance computing (HPC) cluster here. This includes links for things like:

1. Setting up an account and do a CMIC HPC induction (contact cluster-accounts@cs.ucl.ac.uk to request a sesssion)
2. Familiarise yourself with the Sun Grid Engine (SGE)
3. Understand best practices for submitting GPU jobs on the cluster
4. Ensure you have enough scratch space to host your data (!) and otherwise liase with your line manager/superviser to request more scratch space

**Note: To view internal webpages you must have a CS account or request the username and password by emailing cluster-accounts@cs.ucl.ac.uk.**

4. Install a copy of all of the python packages needed for your DL job using ```pip```.  First, prepare a text file that contains a list of all required packages (in the following format: <package_name>==<version>) and save this to your scratch (e.g.,“requirements.txt”):
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

```python3 -m pip install -r requirements.txt --user ***username****```

**Note: It’s worth checking whether the packages required are already installed in /share/apps/python-3.8.5-shared/lib before doing this step.**

## 4. Downloading and organising your data and create a csv file of labels
### 4.1 Importing/downloading data
Getting data onto the cluster is a straight forward process. For example you use ```wget``` for publicly available dataset e.g. for IXI data:    
``` 
wget -cq http://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI/IXI-T1.tar 
wget http://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI/IXI.xls
```

For researchers with approved UK Biobank projects: specific instructions on downloading 'bulk data' using helper programs (e.g. ```ukbfetch```) are available at this link.

You can also import a copy onto the cluster using ```scp``` (example provided below). 
 
### 4.2 Organisation of data directory
I's worth spending some time thinking about the overall structure of your data directory once you've downloaded or imported it onto the cluster. 
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
    
In a seperate ```labels.csv```, you will need to include the ID, label and file pathway for each participant in your sample.
    
### 5. Prepare your script
Put together your python script or Jupyter notebook - Google Colab is a good starting point to do an initial debug of your script. You can download publicly available images for this step e.g., IXI data. Once you’re happy that things generally work – you can import your script to the cluster. 

There are several ways to do this (e.g., ``rsync```), but I tend to use ```scp``` for moving my files between my laptop and the cluster:
### 6. Move your script to cluster
1. Type the following into a new terminal (replace username with your details):
  
  ```ssh -L 2222:comic.cs.ucl.ac.uk:22 username@tails.cs.ucl.ac.uk```
  
2. Then type the following into another terminal (logged into the cluster):
  
  ```scp -P 2222 /Users/ExampleName/Documents/example.py   manaturk@localhost://home/username/scripts```

## 7. Submit bash script to SGE scheduler or request an interactive session
```example.sh```  contains an example script that you can use to submit your DL job

 If you need a short interactive session for debugging you can request using ```qrsh```:
  
 ``` qrsh -l tmem=4G,gpu=true,h_rt=0:30:0 -pe gpu 2```
  
## 8. Evaluating model performance
You can monitor the progress of training through plots of your learning curve of your model

Unfortunately, there isn’t a way to view your plots in the cluster (that I’m aware of), so you will need to copy this onto your local desktop. To do this:
1. Type the following into a new command line terminal (replace username with your details):
  
  ```ssh -L 2222:comic.cs.ucl.ac.uk:22 username@tails.cs.ucl.ac.uk```
2. In another terminal (make sure you are not logged into the cluster):
  
```scp -P 2222 username@localhost:/home/username/plot.pdf /Users/ExampleName/deep_learning_project/```

## 9 Once you are set up on the cluster
Example datasets to run through deep learning tutorials:
IXI: T1-weighted, T2-weighted and DTI images from 600 healthy individuals (ages 20-90)
OASIS-1: T1-weighted images from 416 participants (ages: 18-96, healthy and clinically diagnosed with mild-to-moderate AD)
Loading and preprocessing your images


