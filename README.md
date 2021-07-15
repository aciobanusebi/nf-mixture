# Mixtures of Normalizing Flows

# Paper abstract

Normalizing flows fall into the category of deep generative models. They explicitly model a probability density function. As a result, such a model can learn probabilistic distributions beyond the Gaussian one. Clustering is one of the main unsupervised machine learning tasks. The most common probabilistic approach to solve a clustering problem is via Gaussian mixture models. Although there are a few approaches for constructing mixtures of normalizing flows in the literature, we propose a direct approach and use the masked autoregressive flow as the normalizing flow. We show the results obtained on 2D datasets and then on images. The results contain density plots or tables with clustering metrics in order to quantify the quality of the obtained clusters. Although they usually obtain worse results than other classic models, the 2D results show that more expressive mixtures of distributions can be learned (than the Gaussian mixture models).

# Index words

Machine learning, Clustering, Mixture models, Normalizing flows, Mixtures of normalizing flows

# Code

## .csv files
- smile.csv: smile dataset from https://profs.info.uaic.ro/~pmihaela/DM/datasets%20clustering/

## .py files
- main_large_datasets.py: code for the image datasets
- main_toy_datasets.py: code for the toy 2D datasets
- run_main_large_datasets.py: entry point for running the experiments for the image datasets; it calls main_large_datasets.py on command line with all the possible arguments
- run_main_toy_datasets.py: entry point for running the experiments for the toy 2D datasets; it calls main_toy_datasets.py on command line with all the possible arguments

## .ipynb files
- draft.ipynb: the code in main_large_datasets.py and main_toy_datasets.py to be run in interactive mode; possibly not updated with the latest code in the two .py files
- clustering_exps_Fmnist.ipynb [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aciobanusebi/nf-mixture/blob/main/clustering_exps_Fmnist.ipynb): the experiments with random clustering, k-means, EM/GMM on the F-MNIST dataset
- clustering_exps_Fmnist_PCA.ipynb [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aciobanusebi/nf-mixture/blob/main/clustering_exps_Fmnist_PCA.ipynb): the experiments with random clustering, k-means, EM/GMM on the F-MNIST+PCA dataset
- clustering_exps_cifar10.ipynb [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aciobanusebi/nf-mixture/blob/main/clustering_exps_cifar10.ipynb): the experiments with random clustering, k-means, EM/GMM on the CIFAR10 dataset
- clustering_exps_cifar10_PCA.ipynb [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aciobanusebi/nf-mixture/blob/main/clustering_exps_cifar10_PCA.ipynb): the experiments with random clustering, k-means, EM/GMM on the CIFAR10+PCA dataset
- clustering_exps_mnist.ipynb [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aciobanusebi/nf-mixture/blob/main/clustering_exps_mnist.ipynb): the experiments with random clustering, k-means, EM/GMM on the MNIST dataset
- clustering_exps_mnist_PCA.ipynb [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aciobanusebi/nf-mixture/blob/main/clustering_exps_mnist_PCA.ipynb): the experiments with random clustering, k-means, EM/GMM on the MNIST+PCA dataset
- clustering_exps_mnist5.ipynb [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aciobanusebi/nf-mixture/blob/main/clustering_exps_mnist5.ipynb): the experiments with random clustering, k-means, EM/GMM on the MNIST5 dataset
- clustering_exps_mnist5_PCA.ipynb [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aciobanusebi/nf-mixture/blob/main/clustering_exps_mnist5_PCA.ipynb): the experiments with random clustering, k-means, EM/GMM on the MNIST5+PCA dataset

# Bonus visualization
Circles dataset:

![Circles density plot](https://github.com/aciobanusebi/nf-mixture/blob/main/GIFS/circles.gif)

Moons dataset:

![Moons density plot](https://github.com/aciobanusebi/nf-mixture/blob/main/GIFS/moons.gif)

Pinwheel dataset:

![Pinwheel density plot](https://github.com/aciobanusebi/nf-mixture/blob/main/GIFS/pinwheel.gif)

Smile dataset:

![Smile density plot](https://github.com/aciobanusebi/nf-mixture/blob/main/GIFS/smile.gif)

Two bananas dataset:

![Two bananas density plot](https://github.com/aciobanusebi/nf-mixture/blob/main/GIFS/two_banana.gif)
