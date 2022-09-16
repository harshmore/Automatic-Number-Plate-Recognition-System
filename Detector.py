from unicodedata import name
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.models import load_model
import numpy as np
import pytesseract
from pytesseract import Output

model = load_model('model.h5')
ocr_model = load_model('ocr.h5')
img = cv2.imread('images/Cars3.png')
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
orig_img = img.copy()

img = cv2.resize(img,(224,224))
img = img/255
names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D',
       'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R',
       'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
predictions = model.predict(img.reshape(1,224,224,3))
[xmin,ymin,xmax,ymax] = predictions[0]
h,w = orig_img.shape[:2]

cropped = orig_img[int(ymin*h):int(ymax*h),int(xmin*w):int(xmax*w),:]


# kernel = np.array([[0, -1, 0],
#                    [-1, 5,-1],
#                    [0, -1, 0]])
# image_sharp = cv2.filter2D(src=cropped, ddepth=-1, kernel=kernel)

gray = cv2.cvtColor(cropped,cv2.COLOR_RGB2GRAY)
ret,thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)

_,contours, hierarchy,= cv2.findContours(thresh.copy(), cv2.RETR_LIST , cv2.CHAIN_APPROX_NONE)
text = ''
for c in contours:
    area = cv2.contourArea(c)
    if area >50:
        x,y,w,h = cv2.boundingRect(c)
        new_img = cropped[y:y+h,x:x+w]
        new_img = cv2.resize(new_img,(64,64))
        new_img = new_img/255
        pred = ocr_model.predict(new_img.reshape(1,64,64,3))
        text += names[np.argmax(pred)]

    # cv2.imshow('Bounding Rectangle', orig_image)
text=text[::-1]
print(text[1:])
cv2.imshow('image',cropped)
cv2.waitKey(0)
cv2.destroyAllWindows()
# d = pytesseract.image_to_data(image_sharp, output_type=Output.DICT)
# print(''.join(d['text']))