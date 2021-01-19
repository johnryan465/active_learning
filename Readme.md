# Active Learning

This is a repository for my third year project around active learning.

---
## Layout

Abstracting out the various aspects allows us to work with the different models and experiments in a repeatable and principled way.


### Datasets

These are classes which wrap a dataset and implemement the features which an active learning method may use.


#### Implemented
- MNIST
- CIFAR10
---

### Models

These are machine learning model types which we will be using for our experiment, certain types of active learning methods can only be used with certain machine learning models.


#### Implemented
- vDUQ
- Deep Neural Network
- Bayesian Neural Network
--- 

### Methods

These are active learning methods which we are testing.

#### Implemented

- Random
- BALD
- BatchBALD