library("stringr")

#These first five elements are required for user to define/edit
set.seed(1001)#If you want to replicate your results
dirpathpng<-Sys.glob(file.path("C:/Users/Green Sturgeon/AI_Project/Tile_Images/yolo-tiling-main/yolosliced/ts/MasterImage*.png"))
dirpathTrain<-"C:/Users/Green Sturgeon/AI_Project/TrainYoloV8/train/"
dirpathVal<-"C:/Users/Green Sturgeon/AI_Project/TrainYoloV8/val/"
PercentForTrain<-0.9 # Percentage of pictures you want for the training set, the remaining will go to validation


NumImages<-length(dirpathpng) # total number of images
TrainNum<-NumImages*PercentForTrain #Number of images to train the model on---needed for the sample function

train_png<-sample(dirpathpng,TrainNum,replace=FALSE) #randomly sample the files for the training
val_png<-dirpathpng[!(dirpathpng %in% train_png)] #remaining files go to the validation

####These might require alteration by the user#####
train_png_names<-strsplit(train_png,"ts/") # Split at last folder to get just the file names as second element
trainpngnames<-sapply(train_png_names, "[[",2) # get second element of the string split
val_png_names<-strsplit(val_png,"ts/") # Split at last folder to get just the file names as second element
valpngnames<-sapply(val_png_names, "[[",2) # get second element of the string split

pngTrainDirectory<-paste(dirpathTrain,trainpngnames,sep="") # Make "to" filepath for the png Train files
pngValDirectory<-paste(dirpathVal,valpngnames,sep="") # Make "to" filepath for the png Val files

txtTrainDirectory<-str_replace(pngTrainDirectory,"png","txt") #Make the "to" filepath for the txt Train files
txtValDirectory<-str_replace(pngValDirectory,"png","txt") #Make the "to" filepath for the txt Val files
train_txt<-str_replace(train_png,"png","txt") #Make the "from" filepath for the txt Train files
val_txt<-str_replace(val_png,"png","txt") #Make the "from" filepath for the txt Val files

file.copy(to=pngTrainDirectory,from=train_png) #Copy Train png files to Train folder
file.copy(to=pngValDirectory,from=val_png) #Copy Validate png files to Val folder

file.copy(to=txtTrainDirectory,from=train_txt) #Copy Train txt files to Train folder
file.copy(to=txtValDirectory,from=val_txt) #Copy Validate txt files to Val folder

