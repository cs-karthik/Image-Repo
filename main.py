from ISR.models import RDN, RRDN


import numpy as np
from PIL import Image
import glob
import numpy as np
from numpy import asarray
from ISR.models import RDN, RRDN
import cv2
import os
path='New_Images_Apr26/'
dirct = os.listdir(path)
model = RRDN(weights='gans')
#model = RDN(weights='noise-cancel')
for i in dirct:
    print(i)
    New_Image = Image.open('New_Images_Apr26/'+i).convert('RGB')
    np.array(New_Image).shape
    img = New_Image.resize((512, 512))

#model = RDN(weights='noise-cancel')

# model = RDN(weights='psnr-small')
#model = RDN(weights='psnr-large')
    img.save('Output/compressed.png','PNG', dpi=[300, 300], quality=50)
    compressed_img = Image.open('Output/compressed.png')

    sr_img = model.predict(np.array(compressed_img))
    print("Prediction"+str(i))
    Image.fromarray(sr_img)
    cv2.imwrite("DGAN"+i,sr_img)