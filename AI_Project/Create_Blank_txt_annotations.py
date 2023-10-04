import glob
import os

dirpathpng = glob.glob(r'C:\Users\Green Sturgeon\AI_Project\Annotations\2023\Full Annotations\Census*\Unit*\MasterImage*.png')
print(dirpathpng)
for i in dirpathpng:
    txt_address = i.replace("png", "txt")
    if (os.path.exists(txt_address)==False):
            open(txt_address,"w")