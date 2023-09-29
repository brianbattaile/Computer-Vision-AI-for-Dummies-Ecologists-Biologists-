# AI Computer Vision For Dummies (Biologists/Ecologists)

## Order of Operations
1.  This assumes you have images you want to work on.
2.  Preparing your Computer.  Adds python 3.11 and all needed packages to your computer to do the deep neural network computer vision work.
3.  Annotate Images
4.  Tile Images.   YoloV8 defaults to images 640 x 640 pixels but any size can be used.  My images are much larger AND teh fish I want to detect are relatively small.  So I must break up my images into smaller sizes close to 640 x 640 to train my model.
5.  Train YoloV8  This is re-training the commercial YoloV8 model to work on images that you care about.
6.  Run Model.  using SAHI and YoloV8.txt to create geojson files that mark your objects of interest (ooi's) from a Georeferenced Image.  SAHI cuts up your images into ~640 x 640 pixels, then applies your customized YoloV8 model to find your ooi's
7.  QGIS. Import your images and corresponding geojson files for manual editing into QGIS to remove false positives and add false negatives.
8.  Run "Geojson_to_Yolo_Darknet.py to convert QGIS geojson files into yolo darknet annotation sytle to reread into LabelImg or put back into
	 step 4 to improve your yoloV8 model...yeah!!!

## Preparing Your Computer
Install folders with python and directions...I put my in C:\Users\Green Sturgeon\AI_Project

Install python 3.11  click add to path at install   Installed at C:\Users\Green Sturgeon\AppData\Local\Programs

Install python 3.9--https://github.com/adang1345/PythonWindows/tree/master/3.9.16    python-3.9.16-amd64-full.exe
	Python.org no longer carries it because python is open source and they are stupid.  Unfortunately we need it for LabelImg because label image just closes after trying to do something when run in python 3.11.  Your other option is to use a different annotation program...there are many to choose from
	I created a virtual environment for 3.9 just for the image annotation and a virtual environment for 3.11 for the actual AI stuff.

Install pycharm or your favorite python IDE
The pycharm community edition is free
https://www.jetbrains.com/pycharm/download/?section=windows

Create Local Python Environment for AI_Porject
in cmd-go to the folder you want your local project, for me it is C:\Users\Green Sturgeon\AI_Project and or a python 3.11 virtual environment type

'''bash
python -m venv AIvenv3.11  
'''

or for a particular version
	python3.9 -m venv AIvenv3.9
Activate your environment
cd to C:\Users\Green Sturgeon\AI_Project\AIvenv\Scripts>activate
	or    C:\Users\Green Sturgeon\AI_Project> AIvenv3.9\Scripts\activate
	just type "deactivate" to get out

#Pytorch is needed to do AI stuff...comes with CUDA and cuDNN apparently.
Install pytorch 2.01  https://pytorch.org/get-started/locally/     
#in your virtual environment type
	pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118   

#LabelImg does not work with python greater than 3.9....so download 3.9 and install in the AIvenv3.9 virtual environment
	Install labelImg https://pypi.org/project/sahi/    pip install labelImg

#In your AIvenv3.11 virtual environment	
Install YoloV8  https://pypi.org/project/ultralytics/ pip install ultralytics
	downlaod all five sized models at https://docs.ultralytics.com/models/yolov8/#supported-tasks under the DETECTION heading
Install SAHI  https://pypi.org/project/sahi/  pip install sahi
	# This does the detections by chopping the image into small sections, ~the same size as your tiled images used for training the model, with overlap

For easy random splitting of train, test and validation images
(AIvenv3.11) C:\Users\Green Sturgeon\AI_Project\AIvenv3.11\Scripts>pip install split-folders
