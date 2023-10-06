import os
import pickle
import re
from PIL import Image

def extract_details_from_prediction(obj_pred):
    bbox_values = re.search('<\((.*)\), w', str(obj_pred.bbox)).group(1)
    #bbox_values = re.search(r'<((.*?))', str(obj_pred.bbox)).group(1)
    bbox = list(map(float, bbox_values.split(',')))

    score = float(re.search(r'value: (.*?)[,>]', str(obj_pred.score)).group(1))

    category_values = re.search(r'id: (.*?),', str(obj_pred.category)).group(1)
    category = int(category_values.strip())

    return bbox, score, category

#Path to your images
visuals_path = r"C:\Users\...Your\Folder\Path...\images"
#Path to the corresponding pickle files
pickles_path = r"C:\Users\...Your\Folder\Path...\pickles"
#Path to the output
output_path = r"C:\Users\...Your\Folder\Path...\yoloV8_Output"

for image_file_name in os.listdir(visuals_path):
    image_path = os.path.join(visuals_path, image_file_name)

    # Corresponding pickle file name and path
    pickle_file_name = os.path.splitext(image_file_name)[0] + ".pickle"
    pickle_file_path = os.path.join(pickles_path, pickle_file_name)

    # Check if the pickle file exists
    if not os.path.exists(pickle_file_path):
        print(f"No corresponding pickle file found for image: {image_file_name}")
        continue

    # Load the pickle file data
    with open(pickle_file_path, 'rb') as f:
        data = pickle.load(f)

    # Get the dimensions of the image
    image = Image.open(image_path)
    image_width, image_height = image.size

    # Iterate over predictions and write to file
    for obj_pred in data:
        # Extract details from prediction
        bbox, score, category = extract_details_from_prediction(obj_pred)

        # Calculate the center of the box, width and height
        bbox_center_x = (bbox[0] + bbox[2]) / 2.0
        bbox_center_y = (bbox[1] + bbox[3]) / 2.0
        bbox_width = bbox[2] - bbox[0]
        bbox_height = bbox[3] - bbox[1]

        # Normalize the values
        bbox_center_x /= image_width
        bbox_center_y /= image_height
        bbox_width /= image_width
        bbox_height /= image_height

        # Save to the output file
        output_file_name = os.path.splitext(image_file_name)[0] + ".txt"
        output_file_path = os.path.join(output_path, output_file_name)

        with open(output_file_path, "a") as out_f:  # use "a" to append to file for multiple predictions
            out_f.write(f"{category} {bbox_center_x} {bbox_center_y} {bbox_width} {bbox_height}\n")
