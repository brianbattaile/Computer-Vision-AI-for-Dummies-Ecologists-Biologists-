# If you don't create a blank annotations.txt file with LabelImg for pics with no fish, you need to run this program
# so that the tiling program works if you want to use blank pics to train yolo.

library("stringr")

#dirpathtxt=Sys.glob(file.path("C:\\Users\\Green Sturgeon\\AI_Project\\Annotations\\2023\\Full Annotations\\Census*\\Unit*\\MasterImage*.txt"))
dirpathpng=Sys.glob(file.path("C:\\Users\\Green Sturgeon\\AI_Project\\Annotations\\2023\\Full Annotations\\Census*\\Unit*\\MasterImage*.png"))

for (i in 1:length(dirpathpng))
{
     txt_address<-str_replace(dirpathpng[i],"png","txt")
     if (file.exists(txt_address) == FALSE)
     {
          file.create(txt_address)
     }
}
