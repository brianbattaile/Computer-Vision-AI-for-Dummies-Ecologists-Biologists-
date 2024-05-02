install.packages("exiftoolr")
library("exiftoolr")
install_exiftool()

#Make a lookup table for your classes,
#ObjectsOfInterest is a vector of all the names found in all the tables of all your images in column "10-11+: Repeated Species" in the DIVE table
     #so if you have 6 different objects of interest it will look something like ObjectsOfInterest<-c("Obj1", "Obj2", "Obj3", "Obj4", "Obj5", "Obj6")
#OOInumber is a vector of your class numbers, starting with 0, 
     #so if you have 6 different objects of interest it will look like OOInumber <- c(0,1,2,3,4,5)
ObjectsOfInterest<-c("GS")
OOInumber<-c(0)
classes<-data.frame(ObjectsOfInterest,OOInumber)

DIVE_annotation_dirpath<-Sys.glob(file.path("C:/Your path to Dive annotations/*/*/Image*.csv"))
png_dirpath<-Sys.glob(file.path("C:/Your path to images/*/*/Image*.png"))

for(i in 1:length(DIVE_annotation_dirpath))
{
     tryCatch({
          print(i)
          
          #Writes a blank annotation file if no objects of interest were observed in the image
          if(class(try(read.table(DIVE_annotation_dirpath[i], fill=TRUE, skip=2, sep=","), silent = TRUE)) == "try-error"){
               Yolo<-data.frame(matrix(ncol = 5, nrow = 0))
               yolopath<-paste(strsplit(png_dirpath[i], ".png")[[1]][1],".txt", sep="")
               write.table(Yolo, yolopath, col.names = FALSE, row.names = FALSE)
          }
          DIVETable<-read.table(DIVE_annotation_dirpath[i], fill=TRUE, skip=2, sep=",")
          Metadata<-exif_read(png_dirpath[i])
          
          ObjCenX<-(DIVETable$V4+(DIVETable$V6-DIVETable$V4)/2)/Metadata$ImageWidth
          ObjCenY<-(DIVETable$V5+(DIVETable$V7-DIVETable$V5)/2)/Metadata$ImageHeight
          ObjWidth<-(DIVETable$V6-DIVETable$V4)/Metadata$ImageWidth
          ObjHeight<-(DIVETable$V7-DIVETable$V5)/Metadata$ImageHeight
          Class<-classes[match(DIVETable$V10,classes),2]
          Yolo<-data.frame(Class,ObjCenX,ObjCenY,ObjWidth,ObjHeight)
          yolopath<-paste(strsplit(png_dirpath[i], ".png")[[1]][1],".txt", sep="")
          write.table(Yolo, yolopath, col.names = FALSE, row.names = FALSE)
          }, error=function(e){cat("ERROR :", conditionMessage(e), "\n")})
}
