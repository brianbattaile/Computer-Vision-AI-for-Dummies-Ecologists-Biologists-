import os
import json
import rasterio
from shapely.geometry import shape

#geotiff_path = r'C:\Users\Green Sturgeon\AI_Project\Georeferenced\MasterImage000490.png'
#geojson_annotation = r'C:\Users\Green Sturgeon\AI_Project\Georeferenced\MasterImage000490.geojson'
#output_yolo_path = r'C:\Users\Green Sturgeon\AI_Project\Georeferenced\MasterImage000490.txt'

def convert_coco_to_geojson(geotiff_path, geojson_annotation, output_yolo_path):
    with rasterio.open(geotiff_path) as src:  # open the original image
        transform = src.transform  # a matrix describing the extent of the coco and geolocated pictures
        # print(transform)
        crs = src.crs  # This is the projection type
        # print(crs)
    with open(geojson_annotation) as f:
        json_data = json.load(f)
        #print(json_data)
        #features = json_data['features'][0]['geometry']
        #print(features)

    annotations = []
    for feature in json_data['features']: #loop through annotations
        ID = feature['properties']['category_id']
        #print(feature['geometry']['coordinates'][0][0][0])
        top_left = feature['geometry']['coordinates'][0][2]
        top_right = feature['geometry']['coordinates'][0][1]
        bottom_left = feature['geometry']['coordinates'][0][3]
        cen = shape(feature["geometry"]).centroid
        center = ~transform * (cen.x,cen.y)
        cenx = center[0]/src.shape[1] #normalize between 0 and 1 for yolo format
        ceny = center[1]/src.shape[0] #normalize between 0 and 1 for yolo format
        tl = ~transform * (top_left[0], top_left[1])
        tr = ~transform * (top_right[0], top_right[1])
        bl = ~transform * (bottom_left[0], bottom_left[1])
        width = (tr[0]-tl[0])/src.shape[1]
        height = (bl[1] -tl[1])/src.shape[0]
        line = [ID, cenx, ceny, width, height]
        lines = " ".join(str(e) for e in line)
        #print(lines)
        annotations.append(lines)
    anns = '\n'.join(annotations)
    with open(output_yolo_path, 'w') as f:
        f.write(str(anns))

# geotiff_paths = []
# # os.walk() returns subdirectories, file from current directory and ...
# # follow next directory from subdirectory list recursively until last directory
# for root, dirs, files in os.walk(r"G:\.shortcut-targets-by-id\1_IRSVwxHJ_lLPwru5DzDz_qUrVgKzri2\GS Census\GIS\SSS Pictures\2021\GIS2021\Unit06"):
#     for file in files:
#         if file.endswith(".png"):
#             geotiff_paths.append(os.path.join(root, file))

 # geojson_annotations = []
 for root, dirs, files in os.walk(r"G:\.shortcut-targets-by-id\1_IRSVwxHJ_lLPwru5DzDz_qUrVgKzri2\GS Census\GIS\SSS Pictures\2021\GIS2021\Unit06"):
     for file in files:
         if file.endswith(".geojson"):
             geojson_annotations.append(os.path.join(root, file))


 for geojson_annotation in geojson_annotations:
     filename = geojson_annotation.replace(".geojson", "")  # puts the file where the original png came from
     output_yolo_path = f'{filename}.txt'
     geotiff_path = f'{filename}.png'
     convert_coco_to_geojson(geotiff_path, geojson_annotation, output_yolo_path)
