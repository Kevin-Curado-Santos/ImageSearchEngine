import sys, os
from engines import Resnet50Engine, VGG16Engine, XceptionEngine
import numpy as np

database = [os.path.join("ImageCLEFphoto2008/images/",folder, img) 
            for folder in os.listdir("ImageCLEFphoto2008/images/") for img in os.listdir("ImageCLEFphoto2008/images/" + folder)]
query_img = str(sys.argv[1])

#engines with features non normalized
engine1 = Resnet50Engine(database, False)
engine2 = VGG16Engine(database, False)
engine3 = XceptionEngine(database, False)

#engines with features normalized
engine4 = Resnet50Engine(database, True)
engine5 = VGG16Engine(database, True)
engine6 = XceptionEngine(database, True)

engine_array = [engine1, engine2, engine3, engine4, engine5, engine6]

#relevancy dictionary
relevancy = np.load("relevancy.npz")

percentages = []

for engine in engine_array:
    count = 0
    results = engine.retrieval(f'Topics/SampleImages/{query_img}.jpg')
    for img in results:
        if img[26:-4] in relevancy[query_img]:
            count += 1
    percentages.append(count*100/len(relevancy[query_img]))

for i in range(len(engine_array)):
    print(f'Engine{i+1} : {percentages[i]}')