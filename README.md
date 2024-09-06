<a id="top"></a>
# AI Computer Vision For Dummies (Biologists/Ecologists)
This set of scripts and directions was put together to enable the non-computer scientist to easily build a computer vision AI for object detection in images.  This work was motivated by very large side scan sonar images taken of green sturgeon as part of their stock assessment but can be applied to any other type of still image or video.  Instead of building custom models from scratch, we leverage the expertise of large AI consortiums and their models, and retrain them to find our objects of interest.  Even still, it will take you a day or two to work through this, and that is not including the time it takes to annotate your images, which could take days/weeks/months depending on many factors.

This guide was prepared and tested on a Windows 10 and Windows 11 computer.  Using the CMD line code will most certainly fail on a Mac (but should be pretty close) though the python code should be the same.  Also, for most of this guide, the folder paths will start with the AI_Project folder from this github repository that you will soon download, and everything before that will be replaced by C:\User\...Your\Folder\Path...\ so your path to the AI_Project folder should look like C:\User\...Your\Folder\Path...\AI_Project

## Order of Operations
1.  [Preparing Your Computer](#preparing-your-computer).  Adds python 3.11 and all needed packages to your computer to do the deep learning, convolution neural network computer vision work.
2.  [Annotating Images](#annotating-images).  This assumes you have images you want to work on, but I have provided some of my images to work on, or you can download images from the net.
3.  [Tile Images](#tile-images).   YoloV8 defaults to images 640 x 640 pixels but any size can be used.  My images are much larger AND the fish I want to detect are relatively small.  So I must break up my images into smaller sizes close to 640 x 640 to train my model.
4.  [Train YoloV8](#train-yolov8).  Train the YoloV8 model to work on images and objects of interest (ooi's) that you care about.
5.  [Run Model](#run-model).  Using SAHI and YoloV8.txt to create geojson files that mark your objects of interest from a georeferenced image.  SAHI cuts up your images into ~640 x 640 pixels, then applies your customized YoloV8 model to find your objects of interest.
6.  [QGIS](#qgis). Import your images and corresponding geojson files for manual editing into QGIS to remove false positives and correct false negatives.
7.  [Convert georeferenced annotations back to Yolo format](#convert-georeferenced-annotations-back-to-yolo-format).  Run Geojson_to_Yolo_Darknet.py to convert QGIS geojson files into yolo darknet annotation style to reread into LabelImg or put back into step 4 to improve your yoloV8 model...yeah!!!
8.  [YoloV9 changes](#yolov9-changes).  YoloV9 came out in February 2024, so I've provided all the python and code changes needed to run YoloV9 instead of YoloV8, including with SAHI.

<a id="preparing-your-computer"></a>
## 1. Preparing Your Computer
If you are a github stud, clone this repository (https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository), if not, just go to the green "code" button and download the zip file and install it in your favored location on your computer, I chose to put it here.

`C:\Users\Green Sturgeon\AI_Project`

It's really dumb to have a space in names...don't do what teh person who set up my computer did, Green_Sturgeon would be much better.

### Installing Python
The only reason we are installing python 3.9 is to use an old free annotator.  You might skip to the Annotating Images section and check out DIVE to see if that's what you would prefer to use.  I probably would use DIVE if I were just starting out, but I didn't find a good solution to it not annotating in Yolo Darknet format until later in my journey, but that is all solved now, so if you would prefer to use DIVE, you can skip installing python 3.9 all together.  Otherwise....

We are going to install python 3.11 AND 3.9

Go to https://www.python.org/downloads

Install python 3.11 (This guide was made using 3.11 and broke installing pytorch (later) with 3.12).   When installing at the first window, click "add to path" check box.   I used the default and it installed at C:\Users\Green Sturgeon\AppData\Local\Programs\Python.  You will need to note the location of your install for later.

Install python 3.9.  Unfortunately we need it for LabelImg (See annotation section) because LabelImg just closes after trying to do something when run in python 3.11.  Your other option is to use a different annotation program...there are many to choose from.  I created a virtual environment (explained later) for python 3.9 just for the image annotation and a virtual environment for python 3.11 for the actual AI stuff. 

Python.org no longer allows downloads for 3.9, they must have their reasons, so go to 

https://github.com/adang1345/PythonWindows/tree/master/3.9.16    

and download

python-3.9.16-amd64-full.exe

which will be appropriate for most users, or find the package that works for your system.

### Install your favorite python IDE
The pycharm community edition is free

https://www.jetbrains.com/pycharm/download/?section=windows

At this point, I highly recommend reading or watching a brief tutorial on your chosen editor, they are complicated beasts.  In the least, you will likely need to learn how to assign a python interpreter to your project and learn how to run a script from the editor.  Within pycharm at least, with a script open, you will go into the settings, and under project you will find the python interpreter and set it to the 3.11 python.exe in your 3.11 local environment that we will create next!!!

### Create Local/virtual Python Environment for AI_Project
We are going to make python local/virtual environments, which tends to be good practice because many python versions and packages can interfere with each other, or not work together, as we found with pycharm and 3.12 and LabelImg not working in >3.10, so we can make a local environment to isolate projects that might require different python versions and packages.

#### Navigating in your Command Line Interface
In your Command Prompt (CMD)-navigate to the folder where you downloaded this git hub repository, for me it is C:\Users\Green Sturgeon\AI_Project.  For instance, my CMD opens up to 

`C:\Users\Green Sturgeon>`

So I type 

`C:\Users\Green Sturgeon> cd AI_Project`

cd stands for change directory and AI_Project is the name of the folder I want to get to.   

`C:\Users\Green Sturgeon\AI_Project`

The fast way to do this if you have multiple folders to get through is copy the entire path from your folder explorer and paste that after "cd", for instance

`C:\Users\Green Sturgeon> cd C:\Users\Green Sturgeon\AI_Project\GeoReferenced`

gets me to

`C:\Users\Green Sturgeon\AI_Project\GeoReferenced>`

#### Python 3.11 virtual local environment
In your CMD, type 

`pip install virtualenv`

This imports a package to create virtual environments.

In CMD, navigate to the folder you want the virtual environment to be in, mine is in the top level of the github project you cloned or downloaded as a zip.

`C:\Users\...Your\Folder\Path...\AI_Project`

Now type 
```
python -m virtualenv AIvenv3.11 -p="C:\Users\...Your\Folder\Path...\AppData\Local\Programs\Python\Python311\python.exe"
```
replacing "AIvenv3.11" for your preferred folder name of your virtual environment and "C:\Users\...Your\Folder\Path...\AppData\Local\Programs\Python\Python311\python.exe" with the path to your 3.11 python executable file, which is likely to be similar to mine so ***HOPEFULLY*** you only need to replace the "...Your\Folder\Path..." part of the path.

#### Python 3.9 virtual local environment
Do the same to make a Python 3.9 local environment
```
python -m virtualenv AIvenv3.9 -p="C:\Users\...Your\Folder\Path...\AppData\Local\Programs\Python\Python39\python.exe"
```

#### Activate your virtual local environment
To activate your environment navigate to the Scripts folder in your local environment and type "activate"

`C:\Users\...Your\Folder\Path...\AI_Project\AIvenv3.11\Scripts> activate`

and your CMD prompt will change to this

`(AIvenv3.11) C:\Users\...Your\Folder\Path...\AI_Project\AIvenv3.9\Scripts>`

Just type "deactivate" while in that same Scripts folder to get out of your local environment.  At this point, I found it easiest to have two CMD windows open, one with your python 3.9 local environment activated and the other with your 3.11 python environment activated...and maybe even a third open without a local environment activated.  Keeping track of which is which can take some extra concentration, but having to constantly activate and deactivate virtual environments is *tedious*.

### Install Pytorch 
Pytorch is needed to do AI stuff...comes with CUDA and cuDNN which allow the use of your gpu, instead of your cpu, to do the AI work which is ***MANY*** times faster.  I was not able to install pytorch with Python 3.12, but that might have changed by the time you read this.

Go to

https://pytorch.org/get-started/locally/

And in that orange and grey table select the options that describe your system, I chose the stable version, Widows, Pip, Python and CUDA 11.8.  If for some reason you don't have a supported GPU, you will want to look into using Google Colab if you have serious computing power needs or just use your CPU for small projects. If you have a relatively modern computer with an NVIDIA GPU, your GPU will likely work, regardless, we must try so copy the code the website shows at the bottom of the table and in your python 3.11 virtual environment (doesn't matter what folder you are in CMD) paste it in.  This is what mine looks like. 

`(AIvenv3.11) C:\Users\...Your\Folder\Path...\AI_Project>pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118```   

### Install LabelImg
LabelImg is the program you will use to annotate your training images with, unfortunately, it does not work with python greater than 3.9 so...

In your AIvenv3.9 virtual environment
 
`(AIvenv3.9) C:\Users\...Your\Folder\Path...\AI_Project>`

Type
```
pip install labelImg
```

### Install YoloV8
See https://pypi.org/project/ultralytics/ for more information

In your AIvenv3.11 virtual environment

`(AIvenv3.11) C:\Users\...Your\Folder\Path...\AI_Project>`

type
```
pip install ultralytics
```

### Download the 5 YoloV8 models
Download all five sized models at https://docs.ultralytics.com/models/yolov8/#supported-tasks under the Supported models heading, in the Performance table and Detection(Coco) heading.  Click on the blue YOLOv8n, YOLOv8s etc...  Another place to find these is https://github.com/ultralytics/ultralytics.  A good place to put them is C:\Users\...Your\Folder\Path...\AI_Project\TrainYoloV8

### Install SAHI  
See https://pypi.org/project/sahi/ for more information

In your AIvenv3.11 virtual environment

`(AIvenv3.11) C:\Users\...Your\Folder\Path...\AI_Project>`

Type
```
pip install sahi
```
SAHI does the detections by chopping the image into small sections, ~the same size as your tiled images used for training the model, with overlap so it shouldn't miss anything, more on this later.

### Install split-folders
For easy random splitting of annotated images into train, validation and test(If applicable) groups (more on this later)

`(AIvenv3.11) C:\Users\Green Sturgeon\AI_Project\AIvenv3.11\Scripts>pip install split-folders`

### Install rasterio
This is to help create images that can be uploaded into a GIS program

In your AIvenv3.11 virtual environment

`(AIvenv3.11) C:\Users\...Your\Folder\Path...\AI_Project>`

Type
```
pip install rasterio
```

OK!  That's all you should need as far as programs and packages for you to run the AI on your computer.  So easy!!!

[🔼 Back to top](#top)
<a id="annotating-images"></a>
## 2. Annotating Images

To create an AI model that will detect objects of interest for you, you first must show the AI what it should be looking for.  We do this by annotating images by drawing boxes that surround the objects of interest and labeling those boxes with a class name (like "Fish_Type_1").  Annotation programs help us do this by turning these graphic boxes (and associated labels) into text files that describe where the boxes are in the image, and what class that box belongs to. This is the time-consuming part of creating an AI as many hundreds to thousands of images must be annotated with thousands of boxes surrounding each class of your objects of interest...at least those are the types of numbers one sees thrown around on the internet.  A model that is useful may require much less or much more. 

First order of business, put the images you want to use to train your model in a folder inside

`C:\Users\...Your\Folder\Path...\AI_Project\Annotations`

I have added a few of my images to practice on in AI_Project\Annotations.  One folder has images without annotations and another has the same images with annotations.  I'm assuming you are going through the trouble to learn this because you have your own images you want to work on, but if you don't, there are a number of resources on the net to download images.

I'm going to explain two different ways to annotate images, one is with LabelImg and the other is with DIVE.  DIVE is a more user friendly program than LabelImg and is probably the way to go, but it has an extra step to convert its proprietary native format to and from Yolo Darknet.

To use LabelImg to annotate images,

In your AIvenv3.9 virtual environment

`(AIvenv3.9) C:\Users\...Your\Folder\Path...\AI_Project>`

Type
```
LabelImg
```

And the LabelImg GUI will open.

LabelImg is fairly self explanatory, but go to https://github.com/HumanSignal/labelImg for more information.  In general, you want to enclose your target objects as closely as possible with the annotation/bounding box.  When you create a bounding box (With the "Create RectBox" button), LabelImg will open a window that wants you to fill in the class name that the ooi belongs to.  Each type of object you are interested in identifying will require it's own class label.  You want to save your annotations using the Yolo style (click through the button under the Save icon to find the Yolo option) and save the annotations in the same folder as the images.  LabelImg also automatically creates a classes.txt file in that folder when you annotate an image.  ***If you are opening a previously annotated image, you will need the classes.txt file in the folder with your image***.  Otherwise it will just close when you attempt to edit the previously annotated image.  It is best practice to label ALL of your objects of interest in an image, leaving some out will "confuse" the model and make it less efficient.  Use CTRL plus the mouse wheel to zoom in and out, instead of the dumb buttons.  LabelImg is a very basic annotator and doesn't do more advanced annotations such as masking.  It has its quirks, but surprisingly, I have yet to find a free, simple and easy-to-install alternative that saves in yolo format.

DIVE can be found at https://kitware.github.io/dive/ and you can use either the online version or download it to your desktop.  There is an excellent guide on the website to walk you through how to use it.  The only real issue with DIVE is that it doesn't save in the yolo darknet format so we will need to convert the format it does give.  What we need to do is, once finished with annotating an image, click on the download button at the top, and then click on the "VIAME CSV" button.  Do this even if there aren't any objects of interest in your image, we still need a blank annotations file for each image.  Once finished with all the images you want to annotate and have downloaded the files, open the DIVE_to_YoloDarkNet.R file, change the file paths to your system, and it should produce a Yolo Darknet annotation file for each VIAME.csv file.  Likewise, once we have run the AI to find our objects of interest and it outputs a Yolo Darknet file, we will want the capability to reimport the file back into DIVE to QA/QC and the YoloDarkNet_to_DIVE.R file does that for us.  

[🔼 Back to top](#top)
<a id="tile-images"></a>
## 3. Tile Images

If you have standard sized images, say 1280 x 1280 or smaller (much larger images slow down the processing),  or consistent sized images with objects of interest that are relatively large compared to the size of the image, you will not need to do this step.  The point of tiling the images is to create images the same size for training as you will input in the model for predictions when you go to use the model, for most projects, this probably isn't a problem.  In my case, I can have very large images (~20,000 x 12,000) as well as relatively small images (~1,000 x 1,000) AND the objects of interest in my images are relatively small so breaking up the images into consistent sizes is mandatory for YoloV8 to work.  The convolution part of a convolution neural network reduces the size of the images through "filters" and if your ooi's are to small, then they get lost in the many series of filters of the convolution section.  When we go to implement the model for predictions we will also be cutting the images into a standard size but using the SAHI package to implement the model...again, if you have standard and consistent sized images with relatively large objects of interest, you will not need to use SAHI, in which case just move on to section 4.

### Create blank annotation files if needed
It is sometimes advised to include some images with no objects of interest, and hence no annotations, for the model to work on, these are sometimes termed "background".  If you want to do this, for the images with no annotations you will need a blank annotations.txt file for the tiling program to work.  Open Create_blank_txt_annotations.py in your python IDE, change the file paths to yours and run it, or run it from command.  If you are more comfortable in R you can use Create_blank_txt_annotations.R to do the same thing.

### Tiling the images
This script is from https://towardsdatascience.com/tile-slice-yolo-dataset-for-small-objects-detection-a75bf26f7fa2 .  Open tile_yolo_new_BB.py in your python IDE (or a text editor) and change line 126 file path to the folder where you keep your images.  The folders in lines 120 and 138 must already exist, or the program will tell you they don't and exit.

Activate your python 3.11 virtual environment 

Then navigate to your directory that has the tile_yolo_new.py script and type `python tile_yolo_new_BB.py` to run the python script from CMD

`(AIvenv3.11) C:\Users\...Your\Folder\Path...\AI_Project\Tile_Images\>

Type
```
python tile_yolo_new.py
```

Sliced images with ooi's and annotation files end up in the C:\Users\...Your\Folder\Path...\AI_Project\Tile_Images\yolosliced\ts folder while any tiled images that did not have any ooi's end up in the C:\Users\...Your\Folder\Path...\AI_Project\ile_Images\yolo-tiling-main\yolosliced\ff folder.

After the script is finished, I reannotate the sliced images in LabelImg to clean up any annotated objects of interest that were cut in half by the tiling process.  I kept a split annotation if I could still identify the object of interest as an object of interest and deleted any annotations otherwise. In LabelImg, open "directory" and you can use the Next and Prev Image buttons to quickly go through your tiled images.  Ultimately, this will result in some images with no annotations but still have an annotations.txt file, which is just fine.  Again, ***If you are opening a previously annotated image, you will need the classes.txt file in the folder with your image***.

### Separate your images into Train and Validate categories
To train a Convolution Neural Network(CNN) like YoloV8, you need to split the annotated data into two or three groups.  Usually they are split into ~80-90% Train and ~10-20% Validate and ~10% or so for a Testing category.  YoloV8 doesn't make use of a Test category at this point for the training, so we only want to separate our images into Train and Validate categories.

Use "Separate Train Validate and Test.py" to assign your tiled images and associated annotations into Train, Validate and optional Test folders.  Again, if you are more comfortable with R, you can use "Separate Train and Validate.R", which currently only separates into the Train and Validate groups.  

[🔼 Back to top](#top)
<a id="train-yolov8"></a>
## 4. Train YoloV8

Yolo V8 comes in 5 different model sizes ranging from nano at 3.5 million parameters to extra-large at 68.7 million parameters.  You downloaded these five models already in the section ***"Preparing Your Computer"***.  The difference in size will affect how quickly your model trains and how quickly it works when applied.  If you are working through large numbers of images such as video, or want to implement a fast version for realtime evaluation in video, the nano version may be your best option, if accuracy is paramount and time is no object, the extra-large version may be for you, some experimentation will be required to determine the best model for your application.

There are a large number of argument options for training a YoloV8 model, I will go over a few of the options I found important, but please consult the YoloV8 reference pages https://docs.ultralytics.com/ and specifically https://docs.ultralytics.com/usage/cfg/#train but in general I found this site to be a monster of opaqueness and confusion.  I also found this particular video valuable for explaining the training process and some output options, www.youtube.com/watch?v=gRAyOPjQ9_s "Complete yolo v8 custom object detection tutorial | Windows & Linux"

### Create your .yaml file
This file tells Yolo where your images are, the number of classes you want to train for and the names of those classes.  it's a simple file to create and I've included an example for my work with a single class of ooi's.  It is stored inside C:\Users\...Your\Folder\Path...\AI_Project\TrainYoloV8\

### Run the YoloV8 trainer on a pretrained model
The following command trains a pretrained yoloV8 model, which means YoloV8 comes from the factory trained to detect common everyday things you find in your house or as you are walking around your neighborhood.  (As an aside, it is briefly fun and instructive to run that model on your images and watch the model find chairs, vases and dogs amidst an image of a pristine forest or coral reef.)

In CMD, activate your Python 3.11 virtual environment and navigate to the TrainYoloV8 folder where GSAI_Images.yaml is stored.

`(AIvenv3.11) C:\Users\...Your\Folder\Path...\AI_Project\TrainYoloV8>`

And type
```
yolo task=detect mode=train epochs=300 data=GSAI_Images.yaml model=yolov8x.pt imgsz=640
```

I'll break down the commands used

`task=detect`  Yolo can do a few different computer vision tasks, object detection is just one of them, others include segmentation, pose estimation, tracking, and classification.

`mode=train`  Here we are training the model, When we go to actually use our trained model when we call yolo, we would use mode=predict

`epochs=300`  This has to do with how many times the trainer will run through our images.  We don't want to over or under train our model, thankfully, the trainer tests for these things and will stop at an optimal point.  Advanced users may want to optimize this themselves, but if you are reading this, you are not an advanced user.  My extra-large model stopped at ~120, so 300 was overkill for me, but better to over guess than find your training stopped at your limit resulting with an un-optimized model.  You can restart the training where it left off but I'll leave it to you to figure out how if you need to.

`data=GSAI_Images.yaml`  Points to the .yaml file that is in the folder level that we navigated to in CMD

`model=yolov8x.pt`  Points to one of the five YoloV8 models, also in the folder level that we navigated to in CMD

`imgsz=640`  The size of the images we are using to train the model in pixels, either the x or y dimension

### Run the YoloV8 trainer from an untrained model from "scratch"
Or use this to run a yolov8 model from scratch that has not been pretrained.  The only difference in the call is model=yolov8x.yaml.  I believe yolo grabs this file from the interwebs somewhere... I also did not find much difference in the overall results between this and the pretrained model.
```
yolo task=detect mode=train epochs=10 data=GSAI_Images.yaml model=yolov8x.yaml imgsz=640
```
### Run the trainer from python
	You can use YoloV8_train.py in C:\Users\...Your\Folder\Path...\AI_Project\AI_Project\TrainYoloV8\ and run it from your python IDE
I needed to add the "if __name__ == '__main__':" line to make it work relative to directions in ultralytics and everywhere else....I don't know why

Or to run it from CMD, navigate to the folder it is in 

`(AIvenv3.11) C:\Users\...Your\Folder\Path...\AI_Project\TrainYoloV8>`

and type
```
python YoloV8_train.py
```

### Results of the training
For me, results ended up in C:\Users\...Your\Folder\Path...\AI_Project\TrainYoloV8\runs\detect\train and if you decide to train different models, they will end up in sequentially numbered folders train1, train2 etc., so you should probably rename the folders to something more memorable.  I have also seen the results deposited in python3.11 virtual environment folders.  Inside those folders are a number of diagnostic graphs, images from train and validate with model detections outlined with confidence scores.  The weights folder is your new model and what you will point to when running your model on new images.  I have supplied a model and results trained from the "small" YoloV8 model found C:\Users\...Your\Folder\Path...\AI_Project\TrainYoloV8\runs\detect\train_Small folder.

### Understanding the Training Results
Understanding the diagnostics is a bit opaque and at some levels requires digging into the code or trusting what you might read on-line.  The following is what I've been able to glean from multiple sources.  

Predictions can come in 4 flavors, True Positives (TP), False Positives (FP), True Negatives (TN) and False Negatives (FN).  True positives occur when the model predicted an object of interest correctly.  False Positives occur when the model detected an object of interest in the background that did not exist.  Ture Negatives occur when the computer does not detect an object of interest when none existed in the background.  False Negatives occur when the model did not detect an object of interest when in fact it did exist. There are a number of metrics to summarize the efficiency of the model using these four flavors, but the two most commonly used statistics for computer Vision AI are Recall and Precision.  Recall is 

$$\frac{TP}{(TP + FN)}$$

or in words, the percentage of true predictions of all real objects of interest, or how well do we find the objects of interest. A Higher recall means the model will detect more of our objects of interest, but it is also more likely to find our objects of interest where non exist.
Precision is 

$$\frac{TP}{(TP + FP)}$$

or in words, the percentage of true predictions of all predictions, or the percentage of predictions that are correct.  A high precision means our model is only finding true instances of our object of interest, but at the expense of finding only the obvious ones.  Hence, typically, if we want better Precision, Recall gets worse and vice versa, so there is a Precision Recall curve that shows how these two are co-related.  

![image](https://github.com/brianbattaile/Computer-Vision-AI-for-Dummies-Ecologists-Biologists-/assets/105937466/d3fbf750-bfbd-4bd2-9505-e4d8e840b9f8)

The other rub, is that these are different for different classes.  In our example, we only have one class, green sturgeon, but most applications will have multiple objects of interest, for example, different species of fish, and any given model will be better at identifying some fish than others.  

Another common statistic we are interested in is how close is the bounding box created by the computer model prediction, to our ground truth bounding box we created in the annotations, the closer they are in size and location, the better.  The metric that describes this is the Intersection over Union (IoU) and is defined as 

$$\frac{\text{Area  of  Overlap}}{\text{Area  of  Union}}$$

![image](https://github.com/brianbattaile/Computer-Vision-AI-for-Dummies-Ecologists-Biologists-/assets/105937466/50a67c9b-098b-48bf-b157-93c03b54830c)

Often an IoU of 50% or greater is used as a lower limit to determine if a prediction is a True Positive but that IoU is user defined.  A higher IoU will result in greater Precision but lower Recall.  

Average Precision is defined at a particular IoU, so AP50 is the average precision for an IoU threshold of 50%.  It is more simply defined as the area under the Precision-Recall curve.

$$AP=\int_{r-0}^{1}\ p(r)dr$$
Where r is recall

![image](https://github.com/brianbattaile/Computer-Vision-AI-for-Dummies-Ecologists-Biologists-/assets/105937466/61bcb925-94af-4f37-a86b-91139c57b97b)

In practical terms, it is calculated as a descrete weighted sum of precisions where the weight is the increase in recall from consecutive recall points.

$$AP=\sum_{i=0}^{n-1}(Recall_i-Recall_{i-1}) \times{Precision_i}$$
Where n is the number of precision recall points.

| $$(Recall_i-Recall_{i-1}) \times{Precision_i)}$$ | AP |
|--------------------|---------|
| (0.1-0.0) × 1	     | 0.1     |
| (0.2-0.1) × 1	     | 0.1     |
| (0.3-0.2) × 0.99   | 0.099   |
| (0.4-0.3) × 0.97   | 0.097   |
| (0.5-0.4) × 0.9    | 0.09    |
| (0.6-0.5) × 0.8    | 0.08    |
| (0.7-0.6) × 0.65   | 0.065   |
| (0.8-0.7) × 0.45   | 0.045   |
| (0.9-0.8) × 0.3    | 0.03    |
| (1.0-0.9) × 01     | 0.01    |
| Sum                | 0.716   | 


![image](https://github.com/brianbattaile/Computer-Vision-AI-for-Dummies-Ecologists-Biologists-/assets/105937466/7d715ad6-ad3c-4e28-a815-108177f57231)

The mAP50 (mean AP50) is the average precision averaged over all the different objects of interest classes.  

$$mAP=\frac{1}{c} \sum_{i=1}^c AP_i$$
Where c is the number of classes.  And a final statistic is mAP50-95, which is the mAP averaged over IoU blocks of 0.05 between 0.5 to 0.95. 

The confusion matrix is a matrix in the form of 

$$CM =
\begin{pmatrix}
TP & FP\\
FN & TN 
\end{pmatrix}$$

for a singe class, with columns normalized to 1, (or a percentage) with true on the x axis and predictions on the y axis.  The IoU default threshold is set at 0.45, why not 0.5 I don't know.

In the images that YoloV8 provides of the objects of interest it identifies, along with the bounding box around the object of interest, it includes a confidence score.

There are two types of confidence scores, box confidence and class confidence.  I believe the box confidence is the confidence associated with the bounding boxes created in the training and in predictions when using the model.  

Box confidence is 

$$IoU(pred, truth) \times Pr(Object)$$  

I have not been able to find the equation for calculating $PR(Object)$.

Class confidence is 

$$PR(Class_i|Object) \times PR(Object) \times IoU(pred, truth)$$

Again, unable to find equation for $PR(Class_i|Object)$

A final summary statistic often found is the $F1_{Score}$, which is a measure of the accuracy of the model, and is a summary statistic that weights both precision and recall as a harmonic mean of the two.

$$F1_{Score} = \frac{2\times{Precision}\times{Recall}}{(Precision + Recall)}$$

![image](https://github.com/brianbattaile/Computer-Vision-AI-for-Dummies-Ecologists-Biologists-/assets/105937466/835a5ff9-df7e-4e28-a651-cc5f1bedd436)

A more general form of this equation weights the recall or precision more or less than the other, depending on the needs of the user.

You will also find box_loss, cls_loss and dfl_loss in the graphs.  Equations for these have not been provided by the authors of YoloV8.  In general, the lower these numbers are the better, box loss is a similar concept to IoU.  Cls_loss is the difference between class predictions and ground truth.  Dfl_loss is distribution focal loss and has to do with training for rare classes for which there aren’t many examples and the model attempts to address this.


[🔼 Back to top](#top)
<a id="run-model"></a>
## 5.0 Run Model

### 5.0 Running your basic model
Now that you have trained your model, you can use it to identify things!  To run a basic model from CMD line, activate your 3.11 virtual environment and navigate to

`(AIvenv3.11) C:\Users\...Your\Folder\Path...\AI_Project>`

and type

```
yolo detect predict model="C:\Users\...Your\Folder\Path...\AI_Project\TrainYoloV8\runs\detect\train_Small\weights\best.pt" source="C:\Users\...Your\Folder\Path...\AI_Project\Georeferenced" imgsz=640 save_txt=True
```

`source="C:\Users\...Your\Folder\Path...\AI_Project\Georeferenced"` Is where you are keeping your images you want to analyze

`save_txt=True`  Simply states to save the model predicted image annotations as a .txt file

Which will save labeled images and a yolo_darknet annotation text file in the folder that you are in at the CMD.  You can then read back into LabelImg to fix False Positives and False Negatives by hand....Hey, no one said this was perfect.  But again, please see https://docs.ultralytics.com/usage/cfg/#predict for a list and brief description of the possible arguments.

### 5.1 Using SAHI and your newly trained YoloV8 model
SAHI stands for Slicing Aided Hyper Inference and is designed to find relatively small objects within larger images.  For my use, this is mandatory because of my large, inconsistently sized images with relatively small ooi's, and because we are using this, necessitated our tiling the images in section 3.  See https://docs.ultralytics.com/guides/sahi-tiled-inference/ and https://github.com/obss/sahi.

In your python 3.11 virtual environment from the CMD line

`(AIvenv3.11) C:\Users\...Your\Folder\Path...\AI_Project\Images>

type

```
sahi predict --model_path  "C:\Users\...Your\Folder\Path...\AI_Project\TrainYoloV8\runs\detect\train_Small\weights\best.pt" --model_type yolov8 --source "C:\Users\...Your\Folder\Path...\AI_Project\Georeferenced" --slice_height 640 --slice_width 640 --overlap_height_ratio 0.2 --overlap_width_ratio 0.2 --visual_bbox_thickness 1 --visual_hide_labels TRUE --export_pickle TRUE
```

As an aside...Unfortunately, SAHI does not output a conveniently formatted .txt file with annotations in the yolo format that we can easily import into LabelImg.  The "pickle" file is the .txt file with the annotations but in an unreadable format.  If you want to turn this into a readable file, in your 3.11 virtual environment

type

```
python
import pandas as pd
```
If you don't have pandas it will throw an error, in which case just do this, type

```
exit()
```

to exit python and then type

```
pip install pandas
```

Then rerun the import 

```
python
import pandas as pd
```
Then type
```
unpickled_data=pd.read_pickle(r"C:\Users\...Your\Folder\Path...\AI_Project\Images\runs\predict\exp\pickles\Your_Image_Name.pickle")
print(unpickled_data)
```

to save as readable text file, type
```
import pprint
with open(r"C:\Users\...Your\Folder\Path...\AI_Project\Test\runs\predict\exp\pickles\Your_Image_Name.txt", "a") as f:
	pprint.pprint(unpickled_data, stream=f)
```
You might need to press enter a couple times for it to take...don't know why.  Also, python is sensitive to indentations, so I have found python to be finicky about just copying and pasting the above lines.

To exit python
```
exit()
```
Still, the dumb thing isn't in yolo format and can't be imported into LabelImg.  The Extract_Details_From_Predictions.py script will turn the pickle files into yolo darknet so they can be read back into LabelImg.  You will need to change the folder paths in lines 19, 21 and 23. 

Back to Using SAHI...

As with the yolo call to train a model, there are a huge number of optional arguments.  See https://docs.ultralytics.com/guides/sahi-tiled-inference/#standard-inference-with-yolov8 for further details.

`model_path` is the path to one of the *.pt files in the weights folder.  You should use the best.pt model unless you're an advanced user, but you are reading this so you aren't advanced...just like me.

`source` is your path to the images you want your yoloV8 model to do predictions on

`slice_height` and `slice_width` are the dimensions in pixels you want SAHI to slice your images into...should be the same dimensions you used in part 3 Tiling Images and part 4 Train YoloV8.

`visual_hide_labels`  I didn't want to see the confidence scores or because I only had one class, there was no ambiguity there.  Defaults to False

`visual_bbox_thickness`  This is the thickness of the line of the bounding box that surrounds your ooi's, I prefered a thinner line so it didn't obstruct the image when I went back to correct the model predictions.

### 5.2 Georeference.py (SAHI and YoloV8) on georeferenced images and QGIS

This is a rather specialized section that won't apply to the majority of investigators.  Our images are georeferenced so we want the images and predicted bounding boxes to be georeferenced as well so we can manipulate them in a GIS program such as QGIS, instead of using LabelImg. GeoreferenceBB.py does this using SAHI and YoloV8.  This is from https://github.com/obss/sahi/discussions/870 and all credit goes to the author.  This works on a georeferenced .tif file (geotif) or a .png with associated .xml file that contains georeferencing.  This creates a geojson file of the predicted bounding boxes associated with the image which can be opened in GIS along with the image.  Before running it, you need to change the file paths on rows 59 and 66.  The path on line 59 will find every instance of a .png in the directories and subdirectories below it and will perform the predictions on all these images.  Also, the script is currently set up for .png files, if you are using .tif files, you will need to change lines 61 and 81 from "png" to "tif".  The predictions will be saved as a .geojson file in the same directory as the image with the same name as the image.  You can run this file from your python IDE or from the CMD activate your 3.11 virtual environment and navigate to the folder the python script is in

`(AIvenv3.11) C:\Users\...Your\Folder\Path...\AI_Project\Georeferenced>`

and type

```
python GeoReference.py
```

[🔼 Back to top](#top)
<a id="qgis"></a>
## 6. QGIS

The following applies to manipulating the bounding boxes within the freeware QGIS.  Any Computer Vision model is not going to be perfect, and by importing into QGIS you can correct the False Negatives and False Positives.  

For importing the .geojson into qgis, we need to create the default style, within QGIS go to Project>Properties and click on Default Styles.  
Under default symbols, change fill to outline red or you favorite color

Click on style manager, click on line, click on simple red line, change color or width or whatever.

Edit, click on pencil when layer is selected, use polygon tool to add create a box for a missed ooi, delete polygons with select features tool, adjust polygons with vertex tool for detections with poor fitting bounding boxes.

View>Toolbars>Shape Digitizing Toolbar  Then use "add rectangle from extent" to add bounding boxes.

Save as .geojson file "non newline" type, which can be reconverted to Yolo annotation format in next section for adding to the images to retain the model on a larger data set for next time.  It's a never ending iterative process over time.

[🔼 Back to top](#top)
<a id="convert-georeferenced-annotations-back-to-yolo-format"></a>
## 7 Convert georeferenced annotations back to Yolo format

Run "Geojson_to_Yolo_Darknet.py to convert QGIS .geojson files into yolo darknet annotation sytle.  This is essentially a reverse engineered back transform of the Georeferenced.py script that made coco (Coco is a data set of images of everyday items used to train and benchmark AI computer vision models) formatted annotations (which are based on x-y coordinates of the image in pixels) and turned them into georeferenced coordinates based on the projection of the georeferenced image.  So Geojson_to_Yolo_Darknet.py takes georeferenced annotations and turns them into x-y image pixel coordinates but using the yolo darknet annotation format instead of the coco format.  If you are confused, it's not your fault, it's a relatively young industry and things are not yet standardized, so everyone is doing something different.  Regardless, now you can retrain your model with the new annotations and images that your model helped you identify.  Again, you can run this from your Python IDE or the CMD ***BUT*** you will need to alter the path in line 55 and depending on if you are using .png or .tif files, line 64 as well.

[🔼 Back to top](#top)
<a id="YoloV9-changes"></a>
## 8 YoloV9

1.  Create a new python 3.11 virtual environment for yolov9, I'll repeat the directions from above here for convenience.
   In CMD, navigate to the folder you want the virtual environment to be in, mine is in the top level of the github project you cloned or downloaded as a zip.

`C:\Users\...Your\Folder\Path...\AI_Project`

Now type 
```
python -m virtualenv YoloV9venv3.11 -p="C:\Users\...Your\Folder\Path...\AppData\Local\Programs\Python\Python311\python.exe"
```
replacing "YoloV9venv3.11" for your preferred folder name of your virtual environment and "C:\Users\...Your\Folder\Path...\AppData\Local\Programs\Python\Python311\python.exe" with the path to your 3.11 python executable file, which is likely to be similar to mine so ***HOPEFULLY*** you only need to replace the "...Your\Folder\Path..." part of the path.

2.  I had issues with a component of the pytorch version I installed for yoloV9, so I had to get an older version for this virtual environemnt, I went to
    https://pytorch.org/get-started/previous-versions/ and eventually found a version that worked for me, which was the following.
   
   ```
   pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu118
   ```

   In the Command, activate your new yolov9 virtual environment and paste that code

3.  Next we want to clone the yoloV9 github repository into the virutal environment, this is different from the "pip" install of the YoloV8 highlighting the different approach of the different designers of the two models.  Go to

   `(AIvenv3.11) C:\Users\...Your\Folder\Path...\AI_Project\AIvenv3.11YoloV9\Scripts>`
   
   and type
   
```
git clone https://github.com/WongKinYiu/yolov9.git
```

   then run the requirements.txt file which will install all the necessary packages for YoloV9

   ```
   pip install -r requirements.txt
   ```
4.  Download the model weights, currently N, S and M are not available, only C and E.  Go to
    https://github.com/WongKinYiu/yolov9/releases/tag/v0.1
    or to the blue models under Evaluation on the main github page, click on the model weights and then download.  I then moved them to a folder

    `(AIvenv3.11) C:\Users\...Your\Folder\Path...\AI_Project\TrainYoloV9`

5.  Train the model.
   First you need to change the number of classes, (the number of different objects of interest that you want to find) in the "cfg*.yaml file.  They should be installed here.

  `(AIvenv3.11) C:\Users\...Your\Folder\Path...\AI_Project\AIvenv3.11YoloV9\Scripts\yolov9\models\detect\yolov9-c.yaml`

  The third line down is "nc: 80  # number of classes" so we need to change that number to our number of classes.  80 is the number that the the model is pretrained on to find from the COCO data set
6.  Now we can run the following code to train your model.

```
python train_dual.py --batch 8 --epochs 500 --img 640 --device 0 --min-items 0 --close-mosaic 15 --data "C:/Users/...Your\Folder\Path.../AI_Project/TrainYoloV9/GSAI_Images.yaml" --weights "C:/Users/...Your\Folder\Path.../AI_Project/TrainYoloV9/yolov9-c.pt" --cfg "C:/Users/...Your\Folder\Path.../AI_Project/AIvenv3.11YoloV9/Scripts/yolov9/models/detect/yolov9-c.yaml" --hyp hyp.scratch-high.yaml
```

Now your model is trained and the results should be in

   `C:/Users/...Your\Folder\Path.../AI_Project/AIvenv3.11YoloV9/Scripts/yolov9/runs/train/`
   
7.  Unfortunately there is a bug and this file that need to be edited.
  
    `(AIvenv3.11) C:\Users\...Your\Folder\Path...\AI_Project\AIvenv3.11YoloV9\Scripts\yolov9\utils\general.py`

  line 903 needs to look like 
  
  ```
  prediction = prediction[0][1]
```
8.  Now we can run the inference on your data!!!!

```
python detect.py --img 640 --conf 0.1 --device 0 --weights "C:/Users/...Your\Folder\Path.../AI_Project/AIvenv3.11YoloV9/Scripts/yolov9/runs/train/exp7/weights/best.pt" --source "C:/Users/...Your\Folder\Path.../AI_Project/Georeferenced/Image.png"
```

9.  SAHI-As of yet, SAHI has not been updated to work with YoloV9, so we need to do it ourselves.  Fortunatley, kadirnar has updated SAHI to work with yoloV7, and the people who make yoloV7 also made yoloV9 so behind the scenes, they work in a similar way.  First if you already installed the original SAHI into your yolov9 virtual environment, we need to uninstall it using

```
pip uninstall sahi
```
after that we will type

```
pip install git+https://github.com/kadirnar/Yolov7-SAHI.git
```

I have added a yolov9 folder to the github and in that we need to take two files and use them to replace the old files.
Add yolov9hub.py to 

`C:\Users\...Your\Folder\Path...\AI_Project\AIvenv3.11YoloV9\Lib\site-packages\sahi\models`

Add auto_model.py to

`C:\Users\...Your\Folder\Path...\AI_Project\AIvenv3.11YoloV9\Lib\site-packages\sahi`

That should make it work.

```
sahi predict --model_path  "C:\Users\...Your\Folder\Path...\AI_Project\AIvenv3.11YoloV9\Scripts\yolov9\runs\train\exp\weights\best.pt" --model_type yolov9hub --source "C:\Users\...Your\Folder\Path...\AI_Project\Georeferenced\Image.png" --slice_height 640 --slice_width 640 --overlap_height_ratio 0.2 --overlap_width_ratio 0.2 --visual_bbox_thickness 1 --export_pickle TRUE
```

In order to do the bulk georeferencing, use the new GeoReference.py in the yolov9 folder, only line 66 has changed, plus you will need to change the file paths of course.  You will also likely need to make sure the python interpreter points to the interpreter in your new yolov9 virtual environment.

That's it to use yolov9!  Cross your fingers!

