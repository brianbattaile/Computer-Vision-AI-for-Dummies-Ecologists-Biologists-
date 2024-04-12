import os
import json
import rasterio
from shapely.geometry import box
from sahi.predict import get_sliced_prediction
from sahi import AutoDetectionModel


def coco_to_geojson_bbox(coco_bbox, transform):
    x_min, y_min, width, height = coco_bbox #defining the variables in the coco_bbox
    x_max = x_min + width
    y_max = y_min + height
    top_left = transform * (x_min, y_max) #Calculate the geolocated bbox limits
    print(x_min, y_max)
    print(~transform * top_left)
    bottom_right = transform * (x_max, y_min)
    return box(top_left[0], top_left[1], bottom_right[0], bottom_right[1])

def convert_coco_to_geojson(geotiff_path, coco_annotations, output_geojson_path):
    with rasterio.open(geotiff_path) as src: #open the original image
        transform = src.transform #a matrix describing the extent of the coco and geolocated pictures
        #print(transform)
        crs = src.crs #This is the projection type
        #print(crs)

    features = []
    for annotation in coco_annotations: #loop through annotations
        bbox = annotation['bbox'] #grabs the bbox "vector" from the coco annotation
        geom = coco_to_geojson_bbox(bbox, transform) #Do the transformation
        feature = { #????? Define the transform as a GIS feature now
            'type': 'Feature',
            'geometry': geom.__geo_interface__,
            'properties': {
                'category_id': annotation['category_id']
            }
        }
        features.append(feature) #append the boxes all into a single file
# Takes the features file and turns it into a geojson file????????
    geojson_data = {
        'type': 'FeatureCollection',
        'features': features,
        'crs': {
            'type': 'name',
            'properties': {
                'name': crs.to_string()
            }
        }
    }

    with open(output_geojson_path, 'w') as f:
        json.dump(geojson_data, f)


# paths of original images
#geotiff_path = r'C:\Users\...Your\Folder\Path...\AI_Project\GeoReferenced\MasterImage000490.png'
geotiff_paths = []
# os.walk() returns subdirectories, file from current directory and ...
# follow next directory from subdirectory list recursively until last directory
for root, dirs, files in os.walk(r"C:\Users\...Your\Folder\Path...\AI_Project\GeoReferenced"):
    for file in files:
        if file.endswith(".png"):
            geotiff_paths.append(os.path.join(root, file))

detection_model = AutoDetectionModel.from_pretrained(
    model_type='yolov8',
    model_path='C:/Users/...Your/Folder/Path.../AI_Project/TrainYoloV8/runs/detect/train_XTRA_LARGE/weights/best.pt',  #yolov8_model_path
    confidence_threshold=0.5,
    device="cuda:0", # or 'cpu'
)

for geotiff_path in geotiff_paths:
    result = get_sliced_prediction(
        geotiff_path,
        detection_model,
        slice_height = 640,
        slice_width = 640,
        overlap_height_ratio = 0.2,
        overlap_width_ratio = 0.2,
    )
    #filename = os.path.basename(geotiff_path).replace(".png", "") # puts the file where the GeoReferenced.py is located
    filename = geotiff_path.replace(".png", "")  # puts the file where the original png came from
    output_geojson_path = f'{filename}.geojson'
    coco_annotations = result.to_coco_annotations()
    convert_coco_to_geojson(geotiff_path, coco_annotations, output_geojson_path)  # writes the gojson
