#!/usr/bin/env python3

################################################################
# IKR project
# Authors: Petr Zubalik, Jan Koci
# Date: 26.4.2018
# Brief: tests the Convolutional neural network on .png files
################################################################
from keras.preprocessing.image import img_to_array, load_img
from keras.models import load_model
import numpy as np
import argparse
import cv2
import os
import math
import sys
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, metavar="dir_path",
                    help="path to directory with .png data to test")
parser.add_argument("--model", type=str, metavar="cnn_model",
                    help="the Convolutional neural network model to use")

args = parser.parse_args()

if (not args.data):
    print("ERROR: missing --data argument")
    parser.print_help()
    exit(1)
if (not args.model):
    print("ERROR: missing --model argument")
    exit(1)

# load cnn model
model = load_model(args.model)

dir_name = args.data

# predict the cnn result on each file and save it to file
with open('image_results.txt', 'w') as fh:
    for f in os.listdir(dir_name):
        if (f.endswith(".png")):
            image = cv2.imread(dir_name + f)
            image = image.astype("float") / 255.0
            image = img_to_array(image)
            image = np.expand_dims(image, axis=0)
            (nt, t) = model.predict(image)[0]
            if (t == 1):
                t = 0.9999
            if (t == 0):
                t = 0.0001
            logit = math.log(t/(1-t)) * 1000

            fh.write("{0} {1:.4f} {2}\n".format(f.split('.')[0], logit,
                                            1 if t > nt else 0))
