import shutil
import os, sys, random
import xml.etree.ElementTree as ET
from glob import glob
import pandas as pd
from shutil import copyfile
import pandas as pd
from sklearn import preprocessing, model_selection
import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np
import os
from sklearn.model_selection import train_test_split
from preprocess import preprocess


preprocess()

!pip install ultralytics

from ultralytics import YOLO

# Load a pre-trained YOLOv8 nano model
model = YOLO("yolov8n.pt")

# Train the model using the specified dataset and number of epochs
results = model.train(data="helmet.yaml", epochs=2)
