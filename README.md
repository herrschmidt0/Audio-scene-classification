# Audio-scene-classification
## Overview
Thesis work. Research on audio scene classification and dimensionality reduction possibilities of time-frequency representations.

A GUI is implemented with PyQt5, that makes it possible to test and tune the methods described.

## Theory
The procedure consists of multiple steps:
- Feature extraction: conversion of samples to a time-frequency representation. Methods: STFT, log scaled STFT, mel scale, MFCC.
- Dimensionality reduction: converting the representation above to smaller dimensionality. Methods: PCA, t-SNE, Isomap, SOM (Self Organizing Map).
- Classification or Clustering (CNN or some other method)
