# easyocr2
  
import pandas as pd
import numpy as np
import os
import glob
import easyocr
import cv2
from joblib import Parallel, delayed
from tqdm import tqdm
import matplotlib.pyplot as plt
#%matplotlib inline

input_path = 'C:/Users/gahoh/Desktop/test/'
train = pd.read_csv(input_path+'test.csv')
#test = pd.read_csv(input_path+'test.csv')
#display(train,test)

reader = easyocr.Reader(['en'])
result = reader.readtext('Test0001.png')

print(result)

train_image_path = input_path+'C:/Users/gahoh/Desktop/test'
test_image_path = input_path+'C:/Users/gahoh/Desktop/test'

reader = easyocr.Reader(['en'],gpu=True)    

for p in [train_image_path,test_image_path]:
    for i in train['img_path'].head():
        inputPath = p + i
        print (inputPath)
        # show first 5 images
        image = cv2.imread(inputPath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(16,8))
        plt.imshow(image)
        plt.show()
        # show ocr text
        ocr_text = reader.readtext(inputPath)
        sentence = ""
        for o in ocr_text:
            word = o[1] + ' '
            sentence += word
        print ('OCR Text:',sentence)        
