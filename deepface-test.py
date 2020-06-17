# -*- coding: utf-8 -*-
"""
Created on Wed Jun  10 12:37:51 2020

@author: Alexandre
"""

from deepface import DeepFace
from deepface.extendedmodels import Age, Gender, Race, Emotion
import cv2
import matplotlib.pyplot as plt
import json


### FUNCTIONS -----------------------------------------------------------------

def verify(img1_path, img2_path):
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    
    plt.imshow(img1[:, :, ::-1])
    plt.show()
    plt.imshow(img2[:, :, ::-1])
    plt.show()
    
    result = DeepFace.verify(img1_path, img2_path, model_name="VGG-Face")
    print("result: ", result)
    
    verification = result["verified"]
    
    if verification:
        print("They are same !")
    else:
        print("They are not same !")

    
    
def analyze(img_path):
    models = {}
    models["emotion"] = Emotion.loadModel()
    models["age"] = Age.loadModel()
    models["gender"] = Gender.loadModel()
    models["race"] = Race.loadModel()
    
    res_analyze = DeepFace.analyze(img_path, actions = ['age', 'gender', 'race', 'emotion'], models=models)
    
    print("")
    #print("Age: ", res_analyze["age"])
    #print("Gender: ", res_analyze["gender"])
    #print("Race: ", res_analyze["dominant_race"])
    #print("Emotion: ", res_analyze["dominant_emotion"])
    print(json.dumps(res_analyze))


### TESTS ---------------------------------------------------------------------

#verify("images/Obama.jpg","images/img3.jpg")
analyze("images/Obama.jpg")
