# AI Computer Vision For Dummies (Biologists/Ecologists)

## Order of Operations
1.  Preparing your Computer.  Adds python 3.11 and all needed packages to your computer to do the deep neural network computer vision work.
2.  Annotate Images (This assumes you have images you want to work on)
3.  Tile Images.   YoloV8 defaults to images 640 x 640 pixels but any size can be used.  My images are much larger AND the fish I want to detect are relatively small.  So I must break up my images into smaller sizes close to 640 x 640 to train my model.
4.  Train YoloV8  This is training the YoloV8 model to work on images and objects of interest (ooi's) that you care about.
5.  Run Model.  Using SAHI and YoloV8.txt to create geojson files that mark your ooi's from a georeferenced image.  SAHI cuts up your images into ~640 x 640 pixels, then applies your customized YoloV8 model to find your ooi's
6.  QGIS. Import your images and corresponding geojson files for manual editing into QGIS to remove false positives and add false negatives.
7.  Run Geojson_to_Yolo_Darknet.py to convert QGIS geojson files into yolo darknet annotation sytle to reread into LabelImg or put back into step 4 to improve your yoloV8 model...yeah!!!

## 1. Preparing Your Computer
If you are a github stud, clone this repository, if not, just go to green "code" button and download the zip file and install it in your favored location on your computer, I chose to put it here and renaimed it AI_Project.

C:\Users\Green Sturgeon\AI_Project

### Installing Python

Go to https://www.python.org/downloads/
Install python 3.11 (Or the latest version, this guide was made using 3.11).   click "add to path" at install.   I used the default and installed at C:\Users\Green Sturgeon\AppData\Local\Programs

Install python 3.9-- Go to 

https://github.com/adang1345/PythonWindows/tree/master/3.9.16    

and download

python-3.9.16-amd64-full.exe

Python.org no longer allows downloads for 3.9, they must have their reasons.  Unfortunately we need it for LabelImg (See annotation section) because LabelImg just closes after trying to do something when run in python 3.11.  Your other option is to use a different annotation program...there are many to choose from.  I created a virtual environment (explained later) for python 3.9 just for the image annotation and a virtual environment for python 3.11 for the actual AI stuff.

### Install your favorite python IDE
The pycharm community edition is free

https://www.jetbrains.com/pycharm/download/?section=windows

At this point, I highly recommend reading or watching a tutorial on you chosen editor, they are complicated beasts.  In the least, you will likely need to learn how to assign an interpreter to your project and learn how to run a script from the editor.
SETTING UP PYCHARM????

### Create Local Python Environment for AI_Porject

We are going to make python local environments, which tends to be good practice because many python versions and packages can interfer with each other, so we can make a local environment to isolate projects that might require different python versions and packages.  We ran into this problem with LabelImg.

#### Navigating in your Command Line Interface
In your Command Prompt (CMD)-navigate to the folder where you downloaded this git hub repository, for me it is C:\Users\Green Sturgeon\AI_Project.  For instance, my CMD opens up to 

`C:\Users\Green Sturgeon>`

So I type 

`C:\Users\Green Sturgeon> cd AI_Project`

cd stands for change directory and AI_Projecct is the name of the folder I want to get to.   

`C:\Users\Green Sturgeon\AI_Project`

The fast way to do this if you have multiple folders to get through is copy the entire path from your folder explorer and paste that after "cd", for instance

`C:\Users\Green Sturgeon> cd C:\Users\Green Sturgeon\AI_Project\GeoReferenced`

gets you to

`C:\Users\Green Sturgeon\AI_Project\GeoReferenced>`

#### Python 3.11 local environment

and to create a python 3.11 virtual environment type

```
C:\Users\Green Sturgeon\AI_Project>python -m venv AIvenv3.11
```
replacing "AIvenv3.11" for your preferend folder name of your virtual environment

#### Python 3.9 local environment

or for a particular version, and in our case we need a python 3.9 environment

`python3.9 -m venv AIvenv3.9`

#### Activate your local environment

To activate your environment navigate to the Scripts folder in your local environment and type "activate"

`C:\Users\Green Sturgeon\AI_Project\AIvenv3.11\Scripts> activate`

and your CMD prompt will chage to this

`(AIvenv3.9) C:\Users\Green Sturgeon\AI_Project\AIvenv3.9\Scripts>`

just type "deactivate" while in that same Scripts folder to get our of your local environment

### Install Pytorch 
Pytorch is needed to do AI stuff...comes with CUDA and cuDNN which allow the use of your gpu, instead of your cpu, to do the AI work.

Go to

https://pytorch.org/get-started/locally/

Find the CMD line code that fits your system, I chose the stable version, Widows, Pip, Python and CUDA 11.8.  CUDA is required for using your GPU, if for some reason your don't have a supported GPU, you will want to look into using Google Colab or go down the rabbit hole of trying to figure out how to get yours to work. Hopefully (most likely) yours does work, in which case, copy the code the website gives and in your virtual environment (doesn't matter what folder you are in CMD) paste it in.  This is what mine looks like.

`(AIvenv3.9) C:\Users\Green Sturgeon\AI_Project> pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`   

### Install LabelImg
LabelImg is the program you will annotate your training images with, unfortunately, it does not work with python greater than 3.9....so install in the AIvenv3.9 virtual environment    
 
`(AIvenv3.9) C:\Users\Green Sturgeon\AI_Project> pip install labelImg`

### Install YoloV8
See https://pypi.org/project/ultralytics/ for more information

In your AIvenv3.11 virtual environment	

`(AIvenv3.11) C:\Users\Green Sturgeon\AI_Project> pip install ultralytics`
	
downlaod all five sized models at https://docs.ultralytics.com/models/yolov8/#supported-tasks under the DETECTION heading

### Install SAHI  
See https://pypi.org/project/sahi/ for more information

`(AIvenv3.11) C:\Users\Green Sturgeon\AI_Project> pip install sahi`

SAHI does the detections by chopping the image into small sections, ~the same size as your tiled images used for training the model, with overlap so it shouldn't miss anything, more on this later.

### Install split-folders
For easy random splitting of train, test and validation images (more on this later)

`(AIvenv3.11) C:\Users\Green Sturgeon\AI_Project\AIvenv3.11\Scripts>pip install split-folders`

OK!  That's all you should need as far as programs and packages for you to run the AI on your computer.  So easy!!!

## 2. Annotating Images

First, put the images you want to use to train your model in a folder inside

``C:\Users\Green Sturgeon\AI_Project\Annotations`

To use LabelImg to annotate your images, activate your python 3.9 environment then open LabelImg from CMD by

`(AIvenv3.9) C:\Users\Green Sturgeon\AI_Project> LabelImg`

LabelImg is fairly self explanitory but go to https://github.com/HumanSignal/labelImg for more information.  In general, you want to enclose your target objects as closely as possible with the annotation box.  When you create a bounding box, LabelImg will ask for the class that ooi belongs to.  Each type of object you are intersted in identifying will require it's own class label.  You want to save your annotations using the Yolo style and save the annotations in the same folder as the images.  LabelImg also automatically creates a classes.txt file for a project when you annotate an image.  If you are opening a previously annotated image, you will need the classes.txt file in the folder with your images.  It is best practice to label ALL of your objects of interest in an image, leaving some out will "confuse" the model and make it less efficient.  

## 3. Tile Images
If you have standard sized images, say 1280 x 1280 or smaller (much larger images slow down the processing),  or consistent sized images with objects of interest that are relatively large compared to the size of the image, you will not need to do this step.  The point of tiling the images is to create images the same size for training as you will input in the model for predictions when you go to use the model.  In my case, I can have very large images (~20,000 x 12,000) as well as relatively small images (~1,000 x 1,000) AND the objects of interest in my images are relatively small so breaking up the images into consistent sizes is mandatory for YoloV8 to work.  The convolution part of the convolution neural network reduces the size of the images through "filters" and if your ooi's are to small, then they get lost in the many series of filters of the convolution section.  When we go to implement the model for predictions we will also be cutting the images into a standard size but using the SAHI package to implement the model...again, if you have standard and consistent sized images with relatively large objects of interest, you will not need to use SAHI.

### Create blank annotation files if needed
It is good practice to include some images with no objects of interest, and hence no annotations, for the model to work on.  For these images with no annotations  you will need a blank annotations.txt file for the tiling program to work.  Open Create_blank_txt_annotations.py in your python IDE, change the file paths to yours and run it, or run it from command.  If you are more comfortable in R you can use Create_blank_txt_annotations.R to do the same thing.

### Tiling the images

Open tile_yolo_new_BB.py in your python IDE (or a text editor) and change line 126 file path to the folder where you keep your images.  Lines 120 and 138 automatically create folders inside the folder that tile_yolo_new_BB.py is located to save the tiled images.

Activate your python 3.11 virtual environment 

Then navigate to your directory that has the tile_yolo_new_BB.py script and type `python tile_yolo_new_BB.py` to run the python script from CMD

`(AIvenv3.11) C:\Users\Green Sturgeon\AI_Project\Tile_Images\yolo-tiling-main>python tile_yolo_new_BB.py`

Sliced images with ooi's and annotation files end up in the ~Tile_Images\yolo-tiling-main\yolosliced\ts folder while any tiled images that did not have any ooi's end up in the ~Tile_Images\yolo-tiling-main\yolosliced\ff folder.

After the script is finished, I assume we should reannotate the sliced images in LabelImg to clean up any annotated objects of interest that were cut in half by the tiling program.  I kept a split annotation if I could still identify the object of interest as an object of interest and deleted any annotations otherwise. 
In LabelImg, open "directory" and you can use the Next and Prev Image buttons to quickly go through your tiled images.  Ultimately, this will result in some images with no annotations but still have an annotations.txt file, which is just fine.

### Seperate your images into Train, Validate and Test categories
To train a Convolution Neural Network(CNN) like YoloV8, it is best practice to split the annotated data into a few groups.  Usually they are split into 70-80% Train, 10-20% Validate and 10% or so for Testing.  

Use "Seperate Train Validate and Test.py" to assign your tiled images and associated annotations into Train, Validate and Test folders.  Again, if you are more comfortable with R, you can use Seperate Train and Validate.R, but it currently only seperates into the Train and Validate groups.  I"M NOT SURE YET THAT YOLO USES THE TEST GROUP.

## 4. Train VoloV8

Yolo V8 comes in 5 different model sizes ranging from nano at 3.5 million parameters to extra large at 68.7 million parameters.  The difference in size will affect how quickly your model trains and how quickly it works when applied.  If you are working through large numbers of images such as video, or want to implement a fast version for realtime evaluation in video, the nano version may be your best option, if accuracy is paramount and time is no object, the extral large version may be for you, some experimentation will be required to determine the best model for your application.

There are a large number of options for training a YoloV8 model, I will go over a few of the options I found important, but please consult the YoloV8 reference pages https://docs.ultralytics.com/ and specifically https://docs.ultralytics.com/usage/cfg/#train but in gneral I found this site to be a monster of opaqueness and confusion.  I also found this particular video valuable for explaining the training process and some output options, www.youtube.com/watch?v=gRAyOPjQ9_s "Complete yolo v8 custom object detection tutorial | Windows & Linux"

### Create your .yaml file
This file tells Yolo where your images are, the number of classes you want to train for and the names of those classes.  it's a simple file to create and I've included an example for my work with a single class of ooi's.

Activate your Python 3.11 virtual environment and navigate to the TrainYoloV8 folder where GSAI_Images.yaml is stored.

### Run the YoloV8 trainer on a pretrained model
The following command traines a pretrained yoloV8 model, which means YoloV8 comes from the factory trained to detect common everyday things you find in your house or as you are walking around your neighborhood.  (As an asside, it is briefly fun to run that model on your images and watch the model find chairs and dogs amidst an image of a pristine forest or coral reef.)

`(AIvenv) C:\Users\Green Sturgeon\AI_Project\TrainYoloV8>yolo task=detect mode=train epochs=120 data=GSAI_Images.yaml model=yolov8x.pt imgsz=640`

I'll break down the commands used
task=detect  Yolo can do a few different computer vision tasks, object detection is just one of them, others include segmentation, pose estimation, tracking, and classification.

mode=train  Here we are training the model, When we go to actually use our trained model when we call yolo, we would use mode=predict

epochs=300  This has to do with how many times the trainer will run through our images.  We don't want to over or under train our model, thankfully, the trainer tests for these things and will stop at an optimal point.  Advanced users will want to optimize this themselves, but if you are reading this, you are not an advanced user.  My extra large model stopped at ~120, so 300 was overkill for me, but better to overguess than find your training stopped at your limit resulting with an un-optimized model.  You might be able to restart the training where it left off but that's really beyond this tutorial. 

### Run the YoloV8 trainer from an untrained model from "scratch"
Or use this to run a yolov8 model from scratch that has not been pretrained.  The only difference in the call is model=yolov8x.yaml.  I believe yolo grabs this file from the interwebs somewhere... I also did not find much difference in the overall results between this and the pretrained model.
yolo task=detect mode=train epochs=10 data=GSAI_Images.yaml model=yolov8x.yaml imgsz=640

### Run the trainer from python
You can use YoloV8_train.py in the C:\Users\Green Sturgeon\AI_Project\AI_Project\TrainYoloV8\ and run it from your python IDE
I needed to add the "if __name__ == '__main__':" line to make it work relative to directions in ultralytics and everywhere else....I don't know why

Or in the command line, go to the folder it is in and run 

`C:\Users\Green Sturgeon\AI_Project\TrainYoloV8> python YoloV8_train.py`

### Results of the training
Results end up in C:\Users\Green Sturgeon\AI_Project\TrainYoloV8\runs\detect\train and if you decide to train different models, they will end up in sequentially numbered folders train1, train2 etc., so you should probably rename the folders to something more memorable.  Inside those folders are a number of diagnostic graphs, images from train and validate with model detections outlined with confidence scores.  The weights folder is your new model and what you will point to when running your model on new images.

### Understanding the Training Results

Understanding the diagnostics is a bit opaque and at some levels requires digging into the code or trusting what you might read on-line.  The following is what I've been able to glean from multiple sources.  

Predictions can come in 4 flavors, True Positives (TP), False Positives (FP), True Negatives (TN) and False Negatives (FN).  True positives occur when the model predicted an object of interest correctly.  False Positives occur when the model detected an object of interest that did not exist.  Ture Negatives occur when the computer does not detect an ooi when none exist, kinda strange in practice.  False Negatives occur when the model did not detect and ooi when in fact it did exist. There are a number of metrics to summarize the efficency of the model using these four flavors, but the two most commonly used statistics for computer Vision AI are Recall and Precision.  Recall is $\frac{TP}{(TP + FN)}$ or in words, the percentage of true predictions of all real ooi's, or how well do we find the ooi's. Precision is $\frac{TP}{(TP + FP)}$ or in words, the percentage of true predictions of all predictions, or the percentage of predictions that are correct.  The rub is that if we want better Precision, Recall gets worse and vise versa, so there is a Precision Recall curve that shows how these two are co-related.  The other rub, is that these are different for different classes.  Another common statistic we are interested in is how close is the bounding box created by the computer model prediction, to the bounding box we created in the annotations, the closer they are in size and location, the better.  The metric that describes this is the Intersectin over Union and is defined as the $\frac{\text{Area  of  Overlap}}{\text{Area  of  Union}}$.  Often an IoU of 50% or greater is used as a lower limit to determine if a prediction is a True Positive but that IoU is user defined.  A higher IoU will result in greater Precision but lower Recall.  From these statistics, the Average Precision is calculated.  

$$AP=\sum_{i=0}^{n-1}(Recall_i-Recall_{i-1}) Precision_i$$

Average Precision is defined at a prticular IoU, so AP50 is the average precision for an IoU of 50% or greater.  In other words, it is the weighted sum of precisions at each threshold where the weight is the increase in recall...if that helps you.  It is more simply defined as the area under the Precision-Recall curve. The mAP50 (mean AP50) is the average precision averaged over all the different ooi classes.  

$$mAP=\frac{1}{c} \sum_{i=1}^c AP_i$$

And a final statistic is mAP50-95, which is the mAP over IoU blocks of 0.05 between 0.5 to 0.95.  

The confusion matrix is a matrix in the form of 

$$CM =
\begin{pmatrix}
TP & FP\\
FN & TN 
\end{pmatrix}$$

for a singe class, with columns normalized to 1, (or a percentage) with true on the x axis and predictions on the y axis.  The IoU threshold is set at 0.45, why not 0.5 I don't know.

There are two types of confidence scores, box confidence and class confidence.  I believe the box confidence is the confidence associated with the bounding boxes created in the training and in predictions when using the model.  Box confidence is $IoU(pred, truth) \times Pr(Object)$.  I have not been able to find the equation for calculating $PR(Object)$.
Class confidence is $PR(Class_i|Object) \times PR(Object) \times IoU(pred, truth)$.  Again, unable to find equation for $PR(Class_i|Object)$

A final summary statistic often found is the $F1_{Score}$

$$F1_{Score} = \frac{2\times{Precision}\times{Recall}}{(Preciaion + Recall)}$$

## 5.0 Running your basic model

Now that you have trained your model, you can use it to identify things!  To run a basic model from CMD line,

```
yolo detect predict model='C:\Users\Green Sturgeon\AI_Project\Images\runs\detect\train_XTRA_LARGE\weights\best.pt' source='C:\Users\Green Sturgeon\AI_Project\Images\images' imgsz=640 
```
but again, please see https://docs.ultralytics.com/usage/cfg/#predict for a list and brief description of the possible arguments.

## 5.1 Using SAHI and your newly trained YoloV8 model
SAHI stands for Slicing Aided Hyper Inference and is designed to find relatively small objects within larger images.  For my use, this is mandatory because of my large, inconsistently sized images with relatively small ooi's.  See https://docs.ultralytics.com/guides/sahi-tiled-inference/ and https://github.com/obss/sahi.

In your python 3.11 virtual environment from the CMD line

`(AIvenv3.11) C:\Users\Green Sturgeon\AI_Project\Images> sahi predict --model_path  "C:\Users\Green Sturgeon\AI_Project\Images\runs\detect\train_XTRA_LARGE\weights\best.pt" --model_type yolov8 --source "C:\Users\Green Sturgeon\AI_Project\Images\images" --slice_height 640 --slice_width 640 --visual_bbox_thickness 1 --visual_hide_labels TRUE --visual_bbox_thickness 1 --visual_hide_labels TRUE`

As with the yolo call to train, there are a huge number of optional arguments.  See https://docs.ultralytics.com/guides/sahi-tiled-inference/#standard-inference-with-yolov8 for further details.

model_path is the path to one of the *.pt files in the weights folder.  You should use the best.pt model unless your an advanced user, but you are reading this so you aren't advanced...just like me.

source is your path to the images you want your yoloV8 model to do predictions on

--slice_height and slice_width are the dimensions in pixels you want SAHI to slice your images into...should be the same dimensions you used in part 3 Tiling Images and part 4 Train YoloV8.

visual_hide_labels  I didn't want to see the confidence scores or because I only had one class, there was no ambiguity there.  Defaults to False

visual_bbox_thickness  This is the thickness of the line of the bounding box that surrounds your ooi's, I prefered a thinner line so it didn't obstruct the image when I went back to correct the model predictions.

## 5.2 Georeferenced.py (SAHI and YoloV8) on georeferenced images and QGIS

This is a rather specialzed section that won't apply to the majority of investigators.  Our images are georeferenced so we want the images and predicted bounding boxes to be georeferenced as well so we can manipulate them in a GIS program such as QGIS, instead of using LabelImg.  GeoreferencedBB.py does this using SAHI and YoloV8.  This is from https://github.com/obss/sahi/discussions/870 and all credit goes to the author.  This works on a georeferenced .tif file (geotif) or a .png with associated .xml file that contains georeferencing.  This creates a geojson file of the predicted bounding boxes associated with the image which can be opened in GIS along with the image.  You can run this file from your python IDE or from the CMD activate your 3.11 virtual environment and navigate to the folder the python script is in and type

`(AIvenv3.11) C:\Users\Green Sturgeon\AI_Project\Georeferenced> python GeoReferencedBB.py

But before that you need to change the file paths on rows 59 and 66.  Also, the script is currently set up for .png files, if you are using .tif files, you will need to change lines 61 and 81 from "png" to "tif".

The following applies to manipulating the bounding boxes within the freeware QGIS.  Any Computer Vision model is not going to be perfect, and by importing into QGIS you can correct the False Negatives and False Positives.  

For importing the geojson into qgis, we need to create the default style  Go to Project>Properties and click on Default Styles.  

Under default symbols, change fill to outline red or favoriate color

click on style manager, click on line, click on simple red line, change color or width or whatever.

Edit, click on pencil when layer is selected, use polygon tool to add fish, delete polygons with select features tool, adjust polygons with vertex tool

View>Toolbars>Shape Digitizing Toolbar  Then use "add rectangle from extent"

Save as geojson file non newline type...which can be reconverted to Yolo annotation format in next section for adding to the images to retain the model on a larger data set for next time.  It's a never ending iterativ process over time.

## 6 Convert georeferenced annotations back to Yolo format

Run "Geojson_to_Yolo_Darknet.py to convert QGIS geojson files into yolo darknet annotation sytle.  This is essentially a reverse engineered back transform of the Georeferenced.py script that made coco (Coco is a data set of images of everyday items used to train and benchmark AI computer vision models) formatted annotations (which are based on x-y coordinates of the image in pixels) and turned them into georeferenced coordinates based on the projection of the georeferenced image.  So Geojson_to_Yolo_Darknet.py takes georeferenced annotations and turns them into x-y image pixel coordinates but using the yolo darknet annotation format instead of the coco format.  Its confusing as Fuck.  Now you can retrain your model with the new annotations and images that your model helped you identify.  Again, you can run this from your IDE or the CMD.  Again, you will need to alter the path in line 55 and depending on if you are using .png or .tif files, line 64.

