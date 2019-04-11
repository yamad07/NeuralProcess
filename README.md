# Neural Process
This repository is implementation of Neural Process.


Neural Process is a class of neural latent variable models. This models combine the best of Neural Network and Gaussian Process.

## How it works
<a href="https://gyazo.com/10e3b6d8293652998b9459a19725205c"><img src="https://i.gyazo.com/10e3b6d8293652998b9459a19725205c.png" alt="Image from Gyazo" width="300"/></a>


This figure is a graphical model of neural process. Nerual Process has two model, encoder, aggregater, and conditional decoder.
An encoder is parametarised as neural network, and it from input space into representation space that takes in pairs of (x, y) context values. An aggregater that summarises the encoded inputs. A conditional decoder g that takes as input the sam- pled global latent variable z as

## Experiments
<a href="https://gyazo.com/a18ea7463c22780a123b9d438eade6e3"><img src="https://i.gyazo.com/a18ea7463c22780a123b9d438eade6e3.png" alt="Image from Gyazo" width="300"/></a>


In this repository, I implement the few-shot learning experiments using MNIST. This task is a image completation as a 2D regression task. inputs context data point and complete image by predicting luck data point

## Usage
You can run experiments using Docker:
```
docker-compose -f docker/docker-compose-cpu.yml build
docker-compose -f docker/docker-compose-cpu.yml run experiment python3 experiment.py
```
