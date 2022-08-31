import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.models import load_model
import numpy as np
import pytesseract
from pytesseract import Output

model = load_model('model.h5')

img = cv2.imread('images/Cars3.png')
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
orig_img = img.copy()

img = cv2.resize(img,(224,224))
img = img/255

predictions = model.predict(img.reshape(1,224,224,3))
[xmin,ymin,xmax,ymax] = predictions[0]
h,w = orig_img.shape[:2]

cropped = orig_img[int(ymin*h):int(ymax*h),int(xmin*w):int(xmax*w),:]


kernel = np.array([[0, -1, 0],
                   [-1, 5,-1],
                   [0, -1, 0]])
image_sharp = cv2.filter2D(src=cropped, ddepth=-1, kernel=kernel)


plt.imshow(image_sharp)
plt.show()

d = pytesseract.image_to_data(image_sharp, output_type=Output.DICT)
print(''.join(d['text']))