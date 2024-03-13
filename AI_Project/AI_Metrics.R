library(sf)

#My manually counted files were named "FC1-459.shp"
#My AI counted files were named "MasterImage000459XL.geojson"

#####Path to manually counted shapefile and AI counted geojson file
#MAC
dirpathDots<-Sys.glob(file.path("/Your/File/Path.../FC*.shp")) #Mac
dirpathAI_XL<-Sys.glob(file.path("/Your/File/Path.../MasterImage*XL.geojson"))
#PC
dirpathDots<-Sys.glob(file.path("G:/Your/File/Path.../FC*.shp")) #PC
#### For each inference run using one of the 5 different sized models
dirpathAI_XL<-Sys.glob(file.path("G:/Your/File/Path.../MasterImage*XL.geojson"))

####Make the geojson a data frame.
#dirpathDotsT<-as.data.frame(dirpathDots)
dirpathAI_XLT<-as.data.frame(dirpathAI_XL)
#dirpathAI_LT<-as.data.frame(dirpathAI_L)

######Make empty data table
dfXL <- data.frame(matrix(ncol = 4, nrow = length(dirpathDots)))
names(dfXL)<-c("Total GS", "True Positives", "False Positives", "Difference")

for(i in 1:length(dirpathDots))
{
  #####Read in the shape file
  Dots <- st_read(dirpathDots[i])
  #Make sure it's a GIS type point file
  Dots<-st_cast(Dots,to = "POINT")
  ######read in the geojson
  AI_boxesXL <- st_read(dirpathAI_XL[i])
  
  ######Transform manual and AI counted locations into same coordinate reference system
  Dots_m<-st_transform(Dots, crs = 7801)
  AI_boxesXL_m<-st_transform(AI_boxesXL, crs = 7801)
  
  ####Which dots (manual) and squares (AI) intersect?
  intersectXL<-st_intersects(Dots_m,AI_boxesXL_m)
  
  ####Calculate and put into data table the "Total GS", "True Positives", "False Positives", "Difference"
  XL<-c(dim(Dots)[1], dim(as.data.frame(intersectXL))[1], dim(AI_boxesXL_m)[1]-dim(as.data.frame(intersectXL))[1], dim(as.data.frame(intersectXL))[1]-(dim(AI_boxesXL_m)[1]-dim(as.data.frame(intersectXL))[1]))
  dfXL[i,]<-XL
}

colSums(dfXL)

plot(Dots_m$geometry)
plot(AI_boxesXL_m,add=TRUE)
