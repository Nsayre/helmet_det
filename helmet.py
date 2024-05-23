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


from ultralytics import YOLO
import torch
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

# Load the model
model_25 = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')
