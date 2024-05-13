install.packages("exiftoolr")
library("exiftoolr")
install_exiftool()

ObjectsOfInterest<-c("GS")
OOInumber<-c(0)
classes<-data.frame(ObjectsOfInterest,OOInumber)

DIVE_annotation_dirpath<-Sys.glob(file.path("C:/Your path to Dive annotations/*/*/Image*.csv"))
png_dirpath<-Sys.glob(file.path("C:/Your path to images/*/*/Image*.png"))
YoloDN_dirpath<-Sys.glob(file.path("C:/Your path to images/*/*/Image*.txt"))

#DIVE_annotation_dirpath<-Sys.glob(file.path("C:/Users/Green Sturgeon/Desktop/YoloTestFolder/masterimage*.csv"))
#png_dirpath<-Sys.glob(file.path("C:/Users/Green Sturgeon/Desktop/YoloTestFolder/MasterImage*.png"))
#YoloDN_dirpath<-Sys.glob(file.path("C:/Users/Green Sturgeon/Desktop/YoloTestFolder/MasterImage*.txt"))

for(i in 1:length(DIVE_annotation_dirpath))
{
     tryCatch({
          print(i)
          
          #Writes a blank annotation file if no objects of interest were observed in the image
          if(class(try(read.table(YoloDN_dirpath[i], fill=TRUE, sep=","), silent = TRUE)) == "try-error"){
               DIVE<-data.frame(matrix(ncol = 11, nrow = 0))
               DIVETable<-readLines(DIVE_annotation_dirpath[i])
               line2<-c(strsplit(gsub('"', "", DIVETable[2]), ",")[[1]], "", "", "", "", "", "", "", "") 
               DIVE<-rbind(DIVE, line2)
               colnames(DIVE)<-strsplit(DIVETable[1], ",")[[1]]
               DIVEpath<-paste(strsplit(DIVE_annotation_dirpath[i], ".csv")[[1]][1],"-new", ".csv", sep="")
               write.table(DIVE, DIVEpath, row.names = FALSE, sep=",")
          }
          DIVETable<-readLines(DIVE_annotation_dirpath[i])
          Metadata<-exif_read(png_dirpath[i])
          YoloDN<-read.table(YoloDN_dirpath[i], sep=" ")
          
          ID<-seq(from=1,to=dim(YoloDN)[1],by=1)
          VII<-rep(strsplit(png_dirpath[i],"/")[[1]][6],dim(YoloDN)[1]) #####!!!!!!!  The 6 will need to be changed according to the path
          UFI<-rep(0,dim(YoloDN)[1])
          TL_x<-YoloDN$V2*Metadata$ImageWidth-0.5*YoloDN$V4*Metadata$ImageWidth
          TL_y<-YoloDN$V3*Metadata$ImageHeight-0.5*YoloDN$V5*Metadata$ImageHeight
          BR_x<-YoloDN$V2*Metadata$ImageWidth+0.5*YoloDN$V4*Metadata$ImageWidth
          BR_y<-YoloDN$V3*Metadata$ImageHeight+0.5*YoloDN$V5*Metadata$ImageHeight
          DLC<-rep(1,dim(YoloDN)[1])
          TL<-rep(-1,dim(YoloDN)[1])
          RS<-ObjectsOfInterest[(YoloDN$V1+1)]
          CPA<-rep(1,dim(YoloDN)[1])
          
          
          DIVE<-data.frame(ID, VII, UFI, TL_x, TL_y, BR_x, BR_y, DLC, TL, RS, CPA)
          colnames(DIVE)<-strsplit(DIVETable[1], ",")[[1]]
          line2<-c(strsplit(gsub('"', "", DIVETable[2]), ",")[[1]], "", "", "", "", "", "", "", "") 
          DIVE<-rbind(line2,DIVE)
          
          DIVEpath<-paste(strsplit(DIVE_annotation_dirpath[i], ".csv")[[1]][1],"-new", ".csv", sep="")
          write.table(DIVE, DIVEpath, row.names = FALSE, sep=",")
     }, error=function(e){cat("ERROR :", conditionMessage(e), "\n")})
}

