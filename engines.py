import numpy as np
import os
from keras.preprocessing.image import load_img, img_to_array


def euclidean_distance(img, query_img):
    return np.sqrt(np.sum(np.square(img - query_img)))

def chi_square_distance(imgA, imgB, eps=1e-10):
    return 0.5 * np.sum(((imgA - imgB)**2)/(imgA + imgB + eps))    


class Resnet50Engine:
    
    def __init__(self, database, normalize: bool):
        from keras.applications.resnet import ResNet50, preprocess_input
        self.model = ResNet50(include_top=False, pooling='avg')
        self.preprocess_input = preprocess_input
        self.features = {}
        self.normalize = normalize
        self.load(database)
        
    def load(self, database):
        '''
        saves features extrated from the images in a database in a dictionary in the form 'path': [img_features]
        '''
        if self.normalize:
            outputfile = "Features_resnet50_normalized.npz"
        else:
            outputfile = "Features_resnet50.npz"
            
        if not os.path.isfile(outputfile):
            self.features = {}
            for img in database:
                self.features[img] = self.featurize(img)
            np.savez_compressed(outputfile, **self.features)
        else:
            self.features = np.load(outputfile)

    def featurize(self, img_file):
        img = load_img(img_file, target_size=(224, 224))
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = self.preprocess_input(x)
        feats = self.model.predict(x)
        feats = np.concatenate(feats).ravel()
        if self.normalize:
            feats /= np.linalg.norm(feats)
        return feats
    
    def retrieval(self, query_img):
        distances = []
        query_features = self.featurize(query_img) 
                          
        for img in self.features.keys():
            distances.append(chi_square_distance(self.features[img], query_features))
        sorted_distances = sorted(distances)
        
        relevant_imgs = []
        for dist in sorted_distances[:10]:
            index = distances.index(dist)
            relevant_imgs.append(list(self.features.keys())[index])
            
        return relevant_imgs
    
class VGG16Engine:
    
    def __init__(self, database, normalize: bool):
        from keras.applications.vgg16 import VGG16, preprocess_input
        self.model = VGG16(include_top=False, pooling='avg')
        self.preprocess_input = preprocess_input
        self.features = {}
        self.normalize = normalize
        self.load(database)
        
    def load(self, database):
        '''
        saves features extrated from the images in a database in a dictionary in the form 'path': [img_features]
        '''
        if self.normalize:
            outputfile = "Features_vgg16_normalized.npz"
        else:
            outputfile = "Features_vgg16.npz"
        if not os.path.isfile(outputfile):
            self.features = {}
            for img in database:
                self.features[img] = self.featurize(img)
            np.savez_compressed(outputfile, **self.features)
        else:
            self.features = np.load(outputfile)

    def featurize(self, img_file):
        img = load_img(img_file, target_size=(224, 224))
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = self.preprocess_input(x)
        feats = self.model.predict(x)
        feats = np.concatenate(feats).ravel()
        if self.normalize:
            feats /= np.linalg.norm(feats)
        return feats
    
    def retrieval(self, query_img):
        distances = []
        query_features = self.featurize(query_img) 
                          
        for img in self.features.keys():
            distances.append(chi_square_distance(self.features[img], query_features))
        sorted_distances = sorted(distances)
        
        relevant_imgs = []
        for dist in sorted_distances[:10]:
            index = distances.index(dist)
            relevant_imgs.append(list(self.features.keys())[index])
            
        return relevant_imgs
      
class XceptionEngine:
    
    def __init__(self, database, normalize: bool):
        from keras.applications.xception import Xception, preprocess_input
        self.model = Xception(include_top=False, pooling='avg')
        self.preprocess_input = preprocess_input
        self.features = {}
        self.normalize = normalize
        self.load(database)
        
    def load(self, database):
        '''
        saves features extrated from the images in a database in a dictionary in the form 'path': [img_features]
        '''
        if self.normalize:
            outputfile = "Features_xception_normalized.npz"
        else:
            outputfile = "Features_xception.npz"
        if not os.path.isfile(outputfile):
            self.features = {}
            for img in database:
                self.features[img] = self.featurize(img)
            np.savez_compressed(outputfile, **self.features)
        else:
            self.features = np.load(outputfile)

    def featurize(self, img_file):
        img = load_img(img_file, target_size=(299, 299))
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = self.preprocess_input(x)
        feats = self.model.predict(x)
        feats = np.concatenate(feats).ravel()
        if self.normalize:
            feats /= np.linalg.norm(feats)
        return feats
    
    def retrieval(self, query_img):
        distances = []
        query_features = self.featurize(query_img)  
                         
        for img in self.features.keys():
            distances.append(chi_square_distance(self.features[img], query_features))
        sorted_distances = sorted(distances)
        
        relevant_imgs = []
        for dist in sorted_distances[:10]:
            index = distances.index(dist)
            relevant_imgs.append(list(self.features.keys())[index])
            
        return relevant_imgs
    