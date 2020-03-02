# Clustering and Classification of Audio signals

## Overview
Thesis work. Research on audio clustering and classification possibilities, focusing on the applicability of some dimensionality reduction methods. 

## Theory
The procedure consists of multiple steps:
- Feature extraction: conversion of samples to a time-frequency representation. Methods: STFT, log scaled STFT, mel scale, MFCC. Various frequency bin numbers are tested.
- Reshaping these spectrograms in a way that it is possible to perform dimensionality reduction on them.
- Applying dimensionality reduction. Project to 2D in case of clustering, and higher dimensions in case of classification. Methods: PCA, t-SNE, Isomap, SOM (Self Organizing Map).
- Classification or Clustering

Pipeline:
![image](https://user-images.githubusercontent.com/33284371/75680257-7c153780-5c91-11ea-8699-2b55cf6a9c0f.png)

## GUI
There's also a GUI implemented with PyQt5, that makes it possible to test and tune the methods described.

## Papers
Some inspirational papers and articles are uploaded in the 'papers' folder.
