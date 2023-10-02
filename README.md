# AI Computer Vision For Dummies (Biologists/Ecologists)

## Order of Operations
1.  Preparing your Computer.  Adds python 3.11 and all needed packages to your computer to do the deep neural network computer vision work.
2.  Annotate Images (This assumes you have images you want to work on)
3.  Tile Images.   YoloV8 defaults to images 640 x 640 pixels but any size can be used.  My images are much larger AND teh fish I want to detect are relatively small.  So I must break up my images into smaller sizes close to 640 x 640 to train my model.
4.  Train YoloV8  This is re-training the commercial YoloV8 model to work on images that you care about.
5.  Run Model.  using SAHI and YoloV8.txt to create geojson files that mark your objects of interest (ooi's) from a Georeferenced Image.  SAHI cuts up your images into ~640 x 640 pixels, then applies your customized YoloV8 model to find your ooi's
6.  QGIS. Import your images and corresponding geojson files for manual editing into QGIS to remove false positives and add false negatives.
7.  Run "Geojson_to_Yolo_Darknet.py to convert QGIS geojson files into yolo darknet annotation sytle to reread into LabelImg or put back into
	 step 4 to improve your yoloV8 model...yeah!!!

## 1. Preparing Your Computer
Install folders with python and directions...I put mine in C:\Users\Green Sturgeon\AI_Project

### Installing Python

Install python 3.11  click "add to path" at install.   I installed at C:\Users\Green Sturgeon\AppData\Local\Programs

Install python 3.9-- Go to 

https://github.com/adang1345/PythonWindows/tree/master/3.9.16    

and download

python-3.9.16-amd64-full.exe

Python.org no longer allows downloads for 3.9, they must have their reasons.  Unfortunately we need it for LabelImg (See annotation section) because LabelImg just closes after trying to do something when run in python 3.11.  Your other option is to use a different annotation program...there are many to choose from.  I created a virtual environment for 3.9 just for the image annotation and a virtual environment for 3.11 for the actual AI stuff.

### Install your favorite python IDE
The pycharm community edition is free

https://www.jetbrains.com/pycharm/download/?section=windows

### Create Local Python Environment for AI_Porject
in cmd-go to the folder you want your local project in, for me it is C:\Users\Green Sturgeon\AI_Project and or a python 3.11 virtual environment type

```
python -m venv AIvenv3.11
```
replacing "AIvenv3.11" for your preferend folder name

or for a particular version

`python3.9 -m venv AIvenv3.9`

To activate your environment use cd in the CMD to get into the Scripts folder in your local environment

`C:\Users\Green Sturgeon\AI_Project\AIvenv>  cd Scripts

`C:\Users\Green Sturgeon\AI_Project\AIvenv\Scripts>`

and type 

`activate`

or

`C:\Users\Green Sturgeon\AI_Project> AIvenv3.9\Scripts\activate`

just type

`(AIvenv3.9) C:\Users\Green Sturgeon\AI_Project\AIvenv3.9\Scripts> deactivate`

to get out of your local environment

### Install Pytorch 
Pytorch is needed to do AI stuff...comes with CUDA and cuDNN which could also be installed seperately.  Go to

https://pytorch.org/get-started/locally/

Find the CMD line code that fits your system, I chose the stable version, Widows, Pip, Python and CUDA 11.8.  CUDA is required for using your GPU, if for some reason your don't have a supported GPU, you will want to look into using Google Colab.  Hopefully (most likely) you do, in which case, copy the code and in your virtual environment (doesn't matter what folder you are in CMD) paste it in. 

`(AIvenv3.9) C:\Users\Green Sturgeon\AI_Project> pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`   
### Install LabelImg
LabelImg is the program you will annotate your training images in, unfortunately, it does not work with python greater than 3.9....so install in the AIvenv3.9 virtual environment    
 
`(AIvenv3.9) C:\Users\Green Sturgeon\AI_Project> pip install labelImg`

### Install YoloV8
See https://pypi.org/project/ultralytics/ for more information

In your AIvenv3.11 virtual environment	

`(AIvenv3.11) C:\Users\Green Sturgeon\AI_Project> pip install ultralytics`
	
downlaod all five sized models at https://docs.ultralytics.com/models/yolov8/#supported-tasks under the DETECTION heading

### Install SAHI  
See https://pypi.org/project/sahi/ for more information

`(AIvenv3.11) C:\Users\Green Sturgeon\AI_Project> pip install sahi`

SAHI does the detections by chopping the image into small sections, ~the same size as your tiled images used for training the model, with overlap so it shouldn't miss anything.

### Install split-folders
For easy random splitting of train, test and validation images

`(AIvenv3.11) C:\Users\Green Sturgeon\AI_Project\AIvenv3.11\Scripts>pip install split-folders`

## 2. Annotating Images
To use LabelImg to annotate your images, activate environment in CMD
`C:\Users\Green Sturgeon\AI_Project> AIvenv3.9\Scripts\activate`
Then open LabelImg from CMD by

`(AIvenv3.9) C:\Users\Green Sturgeon\AI_Project> LabelImg`

In general, you want to enclose your target objects as closely as possible with the annotation box.  LabelImg is fairly self explanitory but go to https://github.com/HumanSignal/labelImg for more information.  Each type of object your are intersted in identifying will require it's own class label.  LabelImg automatically creates a classes.txt file when you annotate an image.  If you are opening a previously annotated image, you will need the classes.txt file in the folder with your images.  It is best practice to label all objects of interest in an image, leaving some out will "confuse" the model and make it less efficient.

## 3. Tile Images
If you have standard sized images, say 1280 x 1280 or smaller (much larger images slow down the processing),  or consistent sized images with objects of interest that are relatively large compared to the size of the image, you will not need to do this step.  The point of tiling the images is to create images the same size for training as you will input in the model for predictions when you go to use the model.  In my case, I can have very large images (~20,000 x 12,000) as well as relatively small images (~1,000 x 1,000) AND the objects of interest in my images are relatively small so breaking up the images into consistent sizes is mandatory for YoloV8 to work.  The convolution part of the convolution neural network reduces the size of the images through "filters" and if your ooi's are to small, then they get lost in the many series of filters of the convolution section.  When we go to implement the model for predictions we will also be cutting the images into a standard size but using the SAHI package to implement the model...again, if you have standard and consistent sized images with relatively large objects of interest, you will not need to use SAHI.

### Create blank annotation files if needed
It is good practice to include some images with no objects of interest, and hence no annotations, for the model to work on.  For these images with no annotations  you will need a blank annotations.txt file for the tiling program to work.  Open Create_blank_txt_annotations.py in your python IDE, change the file paths to yours and run it, or run it from command.  If you are more comfortable in R you can use Create_blank_txt_annotations.R to create blank annotation text files for images with no objects in them.

### Tiling the images
Activate your python 3.11 virtual environment from the CMD 

`C:\Users\Green Sturgeon\AI_Project\AIvenv\Scripts> activate`

Then navigate to your directory that has the tile_yolo_new_BB.py script and type `python tile_yolo_new_BB.py` to run the python script from CMD

`(AIvenv) C:\Users\Green Sturgeon\AI_Project\Tile_Images\yolo-tiling-main>python tile_yolo_new_BB.py`

#Need the classes.names document in the yolosliced folder, this is also the folder where sliced images will be saved
#C:/Users/Green Sturgeon/AI_Project/Annotations/2023/Full Annotations/Census_1/ folder has the annotated images that are going to be sliced in the ts folder

After the script is finished, you should reannotate the sliced images to clean up any annotated objects of interest that were cut in half by the tiling program.  I included a split annotation if I could still identify the object of interest as an object of interest and deleted any annotations otherwise. 
In LabelImg, open "directory" and you can use the Next and Prev Image buttons to quickly go through your tiled images.  Ultimately, this will result in some images with not annotations but still have an annotations.txt file, which is just fine.

### Seperate your images into Train, Validate and Test categories
To train a CNN like YoloV8, it is best practice to split the annotated data into a few groups.  Usually they are split into 70-80% Train, 10-20% Validate and 10% or so for Testing.  

Use  "Seperate Train and Validate.R" in C:\Users\Green Sturgeon\AI_Project to put pics and annotations into folders to train and validate the Yolo model
Or use  but needed to pip install split-folders from CLI

