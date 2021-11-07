# A guide to kick starting a deep learning project on UCL CMIC's HPC cluster

## 1. Introduction
This page provides a list of resources to get you started on projects that aim to apply deep learning models based on 3D medical images acquired using Magnetic Resonance Imaging (MRI). It also includes specific instructions on how to set things up on UCL CMIC HPC cluster for new starters. Note that there is quite an overlap between this page and UCL DRC Neuroimaging Analysis Wiki page on [deep learning](https://wiki.ucl.ac.uk/pages/viewpage.action?pageId=181248279) (written by myself & Sophie Martin).

## 2. General overview of deep learning
Deep learning is a subfield of Machine learning where algorithms are structured into layers of nodes or "neurons", otherwise refer "neural networks" (consisting of an input layer, hidden layers and output layer, see example below). While classical machine learning requires features to already be extracted from the images (e.g., FreeSurfer measures of cortical thickness, surface area and cortical/subcortical volumes), deep learning models are able to **learn the features** directly to the raw images, without human intervention. Deep learning models have led to some of the most accurate predictions..

### 2.1 Some example projects where deep learning models have been applied:
2.1.1 Automated brain segmentation (e.g., SynthSeg)

2.1.2 Building predictive models of biological 'brain age' (e.g., SFCN based on T1-weighted images)

2.1.3 Classifying individuals into disease categories (e.g., healthy, MCI, dementia)

### 2.2 Resources 
While an in-depth introduction is beyond the scope of this page, I would highly recommend checking out several excellent resources that have previously been published on the topic:

2.2.1 Brief introductory article on deep learning

2.2.2 Neural Networks and Deep learning course run by Andrew Ng on Coursera

2.2.3 Deep learning (Goodfellow, et al. 2016)

2.2.4 You may also be able to request to audit the following courses (UCL staff and students only):

MPHY0025: Information Processing in Medical Imaging (contact: James Cole, james.cole@ucl.ac.uk)

MPHY0041: Machine Learning in Medical Imaging (contact: Andre Altmann, a.altmann@ucl.ac.uk)

COMP0090: Introduction to Deep Learning (contact: Yipeng Hu, yipeng.hu@ucl.ac.uk)

### 2.3 Getting started
You can find advice on working with the Computer Science high-performance computing (HPC) cluster here. This includes links for things like:

2.3.1 Setting up an account

2.3.2 Familiarise yourself with the Sun Grid Engine (SGE)

2.3.2 Submitting GPU jobs on the cluster

```To view internal webpages you must have a CS account or request the username and password by emailing cluster-accounts@cs.ucl.ac.uk.```

2.3.3 Install a copy of all python packages needed for your DL job, by preparing a text file that contains a list of all required packages saved on your scratch (e.g.,“requirements.txt”):

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

***Note: It’s worth checking whether the packages required are already installed in /share/apps/python-3.8.5-shared/lib before doing this step.***

### 2.4 Once you are set up on the cluster
Example datasets to run through deep learning tutorials:
IXI: T1-weighted, T2-weighted and DTI images from 600 healthy individuals (ages 20-90)
OASIS-1: T1-weighted images from 416 participants (ages: 18-96, healthy and clinically diagnosed with mild-to-moderate AD)
Loading and preprocessing your images

MONAI: a PyTorch-based, open-source framework for deep learning in healthcare imaging:
Series of videos from a 2-day bootcamp introduction to MONAI (2020)
Various tutorials on 2D and 3D classification and segmentation examples
Deep Learning Tool Kit (DLTK): an open-source deep learning library for medical imaging:
Introduction to Tensorflow and biomedical imaging analysis

Several tutorials covering the basics of reading in images, data augmentation and building a model
