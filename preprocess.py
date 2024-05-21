import shutil
import os, sys, random
import xml.etree.ElementTree as ET
from glob import glob
import pandas as pd
from shutil import copyfile
import pandas as pd
from sklearn import preprocessing, model_selection
import numpy as np
import os
from sklearn.model_selection import train_test_split

def preprocess():
    """Preprocesses VOC2028 data for YOLO training"""

    # Rename all JPG file extensions to be lowercase
    jpg_path = os.path.join("raw_data", "VOC2028", "JPEGImages")
    for filename in os.listdir(jpg_path):
        # Split the filename into the name and extension
        base, ext = os.path.splitext(filename)
        # Check if the extension is not in lowercase
        if ext and not ext.islower():
            # Create the new filename with lowercase extension
            new_filename = base + ext.lower()
            # Get full paths for the old and new filenames
            old_file = os.path.join(jpg_path, filename)
            new_file = os.path.join(jpg_path, new_filename)
            # Rename the file
            os.rename(old_file, new_file)
            print(f'Renamed: {old_file} to {new_file}')

    # Gather all .xml files from raw data annotations
    annot_path = os.path.join("raw_data", "VOC2028", "Annotations", "*.xml")
    annotations = sorted(glob(annot_path))

    # Parse and gather XML annotation data
    df = []
    cnt = 0
    for file in annotations:
        prev_filename = file.split('/')[-1].split('.')[0] + '.jpg'
        filename = str(cnt) + '.jpg'
        row = []
        parsedXML = ET.parse(file)
        for node in parsedXML.getroot().iter('object'):
            helmet = node.find('name').text
            xmin = int(node.find('bndbox/xmin').text)
            xmax = int(node.find('bndbox/xmax').text)
            ymin = int(node.find('bndbox/ymin').text)
            ymax = int(node.find('bndbox/ymax').text)

            row = [prev_filename, filename, helmet, xmin, xmax, ymin, ymax]
            df.append(row)
        cnt += 1

    # Create dataframe
    data = pd.DataFrame(df, columns=['prev_filename', 'filename', 'helmet_or_person', 'xmin', 'xmax', 'ymin', 'ymax'])


    # TODO
    # save CSV file
    # csv_fp = os.path.join("raw_data", "VOC2028", "Annotations", "*.xml")
    # data[['prev_filename','filename', 'helmet_or_dog_person', 'xmin', 'xmax', 'ymin', 'ymax']].to_csv('/Users/iliyask/Helmet_iliyas/helmet_det/output.csv', index=False)

    # TODO remove dogs
    data = data[data["helmet_or_person"] != "dog"]

    # TODO read actual image size
    img_width = 640
    img_height = 480

    def width(df):
      return int(df.xmax - df.xmin)
    def height(df):
      return int(df.ymax - df.ymin)
    def x_center(df):
      return int(df.xmin + (df.width/2))
    def y_center(df):
      return int(df.ymin + (df.height/2))
    def w_norm(df):
      return df/img_width
    def h_norm(df):
      return df/img_height

    # df = pd.read_csv('/Users/iliyask/Helmet_iliyas/helmet_det/output.csv')
    df = data[['prev_filename','filename', 'helmet_or_person', 'xmin', 'xmax', 'ymin', 'ymax']]

    le = preprocessing.LabelEncoder()
    le.fit(df['helmet_or_person'])
    # print(le.classes_)
    labels = le.transform(df['helmet_or_person'])
    df['labels'] = labels

    df['width'] = df.apply(width, axis=1)
    df['height'] = df.apply(height, axis=1)

    df['x_center'] = df.apply(x_center, axis=1)
    df['y_center'] = df.apply(y_center, axis=1)

    df['x_center_norm'] = df['x_center'].apply(w_norm)
    df['width_norm'] = df['width'].apply(w_norm)

    df['y_center_norm'] = df['y_center'].apply(h_norm)
    df['height_norm'] = df['height'].apply(h_norm)

    assert len(set(df['prev_filename'])) == 7581
    assert len(set(df['filename'])) == 7581

    unique_values = df['prev_filename'].unique()

    # 2. Split these unique values into training and testing sets
    df_train1, df_test = train_test_split(unique_values, test_size=0.1, random_state=13, shuffle=True)
    # 3. Create the training and testing DataFrames based on these splits
    df_train1 = df[df['prev_filename'].isin(df_train1)]
    df_test = df[df['prev_filename'].isin(df_test)]


    unique_values_train = df_train1['prev_filename'].unique()

    # 2. Split these unique values into training and testing sets
    df_train, df_val = train_test_split(unique_values_train, test_size=0.1, random_state=13, shuffle=True)
    # 3. Create the training and testing DataFrames based on these splits
    df_train = df_train1[df_train1['prev_filename'].isin(df_train)]
    df_val = df_train1[df_train1['prev_filename'].isin(df_val)]

    # make folders
    if not os.path.isdir("data"):
        os.mkdir('data')
    if not os.path.isdir("data/images"):
        os.mkdir('data/images/')

    if not os.path.isdir("data/images/train"):
        os.mkdir('data/images/train/')
    if not os.path.isdir("data/images/test"):
        os.mkdir('data/images/test/')
    if not os.path.isdir("data/images/val"):
        os.mkdir('data/images/val/')

    if not os.path.isdir("data/labels"):
        os.mkdir('data/labels/')
    if not os.path.isdir("data/labels/train"):
        os.mkdir('data/labels/train/')
    if not os.path.isdir("data/labels/test"):
        os.mkdir('data/labels/test/')
    if not os.path.isdir("data/labels/val"):
        os.mkdir('data/labels/val')
    
    def segregate_data(df, img_path, label_path, train_img_path, train_label_path):
      filenames = []
      for filename in df.filename:
        filenames.append(filename)
      filenames = set(filenames)

      for filename in filenames:
        yolo_list = []

        for _,row in df[df.filename == filename].iterrows():
          yolo_list.append([row.labels, row.x_center_norm, row.y_center_norm, row.width_norm, row.height_norm])

        yolo_list = np.array(yolo_list)
        txt_filename = os.path.join(train_label_path,str(row.prev_filename.split('.')[0])+".txt")
        # Save the .img & .txt files to the corresponding train and validation folders
        np.savetxt(txt_filename, yolo_list, fmt=["%d", "%f", "%f", "%f", "%f"])
        shutil.copyfile(os.path.join(img_path,row.prev_filename), os.path.join(train_img_path,row.prev_filename))

    ## Apply function ##

    # src_img_path = "/Users/iliyask/Helmet_iliyas/helmet_det/VOC2028/JPEGImages"
    # src_label_path = "/Users/iliyask/Helmet_iliyas/helmet_det/VOC2028/Annotations"
    src_img_path = os.path.join("raw_data", "VOC2028", "JPEGImages")
    src_label_path = os.path.join("raw_data", "VOC2028", "Annotations")

    # train_img_path = "/Users/iliyask/Helmet_iliyas/helmet_det/Data/images/train"
    # train_label_path = "/Users/iliyask/Helmet_iliyas/helmet_det/Data/labels/train"
    train_img_path = os.path.join("data", "images", "train")
    train_label_path = os.path.join("data", "labels", "train")

    # test_img_path = "/Users/iliyask/Helmet_iliyas/helmet_det/Data/images/test"
    # test_label_path = "/Users/iliyask/Helmet_iliyas/helmet_det/Data/labels/test"
    test_img_path = os.path.join("data", "images", "test")
    test_label_path = os.path.join("data", "labels", "test")

    # val_img_path = "/Users/iliyask/Helmet_iliyas/helmet_det/Data/images/val"
    # val_label_path = "/Users/iliyask/Helmet_iliyas/helmet_det/Data/labels/val"
    val_img_path = os.path.join("data", "images", "val")
    val_label_path = os.path.join("data", "labels", "val")

    segregate_data(df_train, src_img_path, src_label_path, train_img_path, train_label_path)
    segregate_data(df_test, src_img_path, src_label_path, test_img_path, test_label_path)
    segregate_data(df_val, src_img_path, src_label_path, val_img_path, val_label_path)

    # print("No. of Training images", len(os.listdir(train_img_path)))
    # print("No. of Training labels", len(os.listdir(train_label_path)))

    # print("No. of test images", len(os.listdir(test_img_path)))
    # print("No. of test labels", len(os.listdir(test_label_path)))

    # print("No. of valid images", len(os.listdir(val_img_path)))
    # print("No. of valid labels", len(os.listdir(val_label_path)))

    # breakpoint()

if __name__ == "__main__":
    preprocess()
