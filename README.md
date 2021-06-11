# eyeDo: an Android App for the Visually Impaired that uses a Neural Network to recognize Pedestrian Traffic Light in real-time

You can find the link to the main project here: https://github.com/marcoleino/eyeDo_Android

This repository contains an implementation of an Object Detection network capable to recognize pedestrian traffic lights.

The dataset we used for our project is based on the ones used for the project "ImVisible", explained at this link: https://arxiv.org/abs/1907.09706.

The model is based on the tutorial 'Train your own object detector with Faster-RCNN & PyTorch'.
You can find the tutorial [here](https://johschmidt42.medium.com/train-your-own-object-detector-with-faster-rcnn-pytorch-8d3c759cfc70).
If you want to use neptune for your own experiments, change [api_key_neptune.py](api_key_neptune.py) accordingly.

A complete jupyter notebook can be found [here](training_script.ipynb).

To try the network on pedestrian traffic lights of Bologna use [jupiter_for_inference.ipynb](jupiter_for_inference.ipynb).

