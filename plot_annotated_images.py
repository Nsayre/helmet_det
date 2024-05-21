import os
from glob import glob
import cv2
import xml.etree.ElementTree as ET
from tqdm import tqdm

def draw_rectangles_from_text(image_path, text_path):
    """
    Draws rectangles on an image based on the annotations in a text file.
    Args:
        image_path (str): Path to the image file.
        text_path (str): Path to the text file containing the annotations.
    Returns:
        numpy.ndarray: The image with rectangles drawn on it.
    """
    CLASS_COLORS = {}
    # Load the image
    image = cv2.imread(image_path)
    height, width, _ = image.shape
    # Read the annotations from the text file
    with open(text_path, 'r') as text_file:
        lines = text_file.readlines()
        # Iterate over the annotations
        for line in lines:
            class_name, x_center_norm, y_center_norm, width_norm, height_norm = line.strip().split()
            x_center_norm, y_center_norm, width_norm, height_norm = map(float, [x_center_norm, y_center_norm, width_norm, height_norm])
            # Calculate the bounding box coordinates
            x_center = int(x_center_norm * width)
            y_center = int(y_center_norm * height)
            box_width = int(width_norm * width)
            box_height = int(height_norm * height)
            xmin = max(0, x_center - box_width // 2)
            ymin = max(0, y_center - box_height // 2)
            xmax = min(width - 1, xmin + box_width)
            ymax = min(height - 1, ymin + box_height)
            # Draw the rectangle on the image
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(image, class_name, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 2)
    return image

# Plot original .xml annotated images
def draw_rectangles_from_xml(image_path, xml_path):
    """
    Draws rectangles on an image based on the bounding box coordinates in an XML file.
    Args:
        image_path (str): Path to the image file.
        xml_path (str): Path to the XML file containing the bounding box coordinates.
    Returns:
        numpy.ndarray: The image with rectangles drawn on it.
    """
    # Load the image
    image = cv2.imread(image_path)
    # Parse the XML file
    tree = ET.parse(xml_path)
    root = tree.getroot()
    # Iterate over the bounding boxes in the XML file
    for obj in root.findall('object'):
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        # Draw the rectangle on the image
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
    return image

def main():


    all_jpg_path = os.path.join("raw_data", "VOC2028", "JPEGImages", "*.jpg")
    # Glob all xml files
    annot_path = os.path.join("raw_data", "VOC2028", "Annotations", "*.xml")

    annotations = sorted(glob(annot_path))
    sorted_jpg_fp = sorted(glob(all_jpg_path))
    sorted_xml_fp = sorted(glob(annot_path))
    
    # loop through all images and annotations, saving plotted images
    print("Creating .xml annotated images...")
    for jpg_fp, xml_fp in tqdm(zip(sorted_jpg_fp, sorted_xml_fp)):
        image = draw_rectangles_from_xml(jpg_fp, xml_fp)
        cv2.imwrite('data/plotted_annotations_xml/' + os.path.basename(jpg_fp), image)

    # all_txt_path = os.path.join("data", "VOC2028", "JPEGImages", "*.jpg")
    # print("Creating .txt annotated images...")
    # for jpg_fp, xml_fp in tqdm(zip(sorted_jpg_fp, sorted_xml_fp)):
    #     image = draw_rectangles_from_xml(jpg_fp, xml_fp)
    #     cv2.imwrite('data/plotted_annotations_xml/' + os.path.basename(jpg_fp), image)

if __name__ == "__main__":
    # Make folders for plotted images
    if not os.path.isdir('data/plotted_annotations_xml'):
        os.mkdir('data/plotted_annotations_xml')
    if not os.path.isdir('data/plotted_annotations_txt'):
        os.mkdir('data/plotted_annotations_txt')

    jpg_fp = os.path.join("data", "images", "test", "000402.jpg")
    txt_fp = os.path.join("data", "labels", "test", "000402.txt")
    image = draw_rectangles_from_text(jpg_fp, txt_fp)
    cv2.imwrite('data/plotted_annotations_txt/' + os.path.basename(jpg_fp), image)
    
    # main()
