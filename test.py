import sys
import os

from numpy import result_type
from engines import Resnet50Engine, VGG16Engine, XceptionEngine
from gallery import plot_gallery

database = [os.path.join("ImageCLEFphoto2008/images/",folder, img) 
            for folder in os.listdir("ImageCLEFphoto2008/images/") for img in os.listdir("ImageCLEFphoto2008/images/" + folder)]
query_img1 = "Topics/SampleImages/02/16432.jpg"
query_img2 = "Topics/SampleImages/02/37395.jpg"
query_img3 = "Topics/SampleImages/02/40498.jpg"

engine = XceptionEngine(database, True)
results = engine.retrieval(query_img2)
print(results)
#plot_gallery(query_img1, results, "relevance_query_1.png")