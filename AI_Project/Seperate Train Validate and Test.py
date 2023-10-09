import splitfolders  # or import split_folders

input_folder = r'C:\Users\Green Sturgeon\AI_Project\Tile_Images\yolo-tiling-main\yolosliced\ts'

# Split with a ratio.
# To only split into training and validation set, set a tuple to `ratio`, i.e, `(.8, .2)`.
#Train, val, test
splitfolders.ratio(input_folder, output=r"C:\Users\Green Sturgeon\AI_Project\TrainYoloV8\PyTest",
                   seed=42, ratio=(.9, .1), #For Train, Validate and Test ratio=(0.8, 0.1, 0.1)
                   group_prefix=2) # we have both .png and .txt annotation files


# Split val/test with a fixed number of items e.g. 100 for each set.
# To only split into training and validation set, use a single number to `fixed`, i.e., `10`.
# enable oversampling of imbalanced datasets, works only with fixed
#splitfolders.fixed(input_folder, output="cell_images2",
#                   seed=42, fixed=(35, 20),
#                   oversample=False, group_prefix=2)