# Advanced Deep Learning final project

This folder contains the files corresponding to the final project of the Advanced Deep Learning and Kernel Methods course at the University of Trieste.
The project consists in the implementation of an explicit regularization in an autoencoder loss, that alow for better separation of the latent vectors, with the ultimate idea of generating a sparser latent space.

The main idea is taken from the paper [Auto-Encoder based Data Clustering](./papers/978-3-642-41822-8_15.pdf) from Song C. et al.

All the experiments are conducted on the `FashionMNIST` dataset, and the code is implemented in `PyTorch`. All the code can be found in the [`experiments.ipynb`](./experiments.ipynb) notebook.

The main results are presented in [this](slides.pdf) presentation.