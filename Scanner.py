import cv2
import numpy as np


def mapp(h):
    h = h.reshape((4,2))
    hnew = np.zeros((4,2),dtype = np.float32)

    add = h.sum(1)
    hnew[0] = h[np.argmin(add)]
    hnew[2] = h[np.argmax(add)]

    diff = np.diff(h,axis = 1)
    hnew[1] = h[np.argmin(diff)]
    hnew[3] = h[np.argmax(diff)]

    return hnew

image=cv2.imread("raw2.jpg")   #read in the image
height = image.shape[0]
width = image.shape[1]
image= cv2.resize(image,(width // 5, height // 5)) #resizing because opencv does not work well with bigger images #resizing because opencv does not work well with bigger images
orig=image.copy()

gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)  #RGB To Gray Scale
cv2.imshow('Title',gray)

blurred=cv2.GaussianBlur(gray,(5,5),0)  #(5,5) is the kernel size and 0 is sigma that determines the amount of blur
cv2.imshow("Blur",blurred)

edged=cv2.Canny(blurred,30,50)  #30 MinThreshold and 50 is the MaxThreshold
cv2.imshow("Canny",edged)


contours,hierarchy=cv2.findContours(edged,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)  #retrieve the contours as a list, with simple apprximation model
contours=sorted(contours,key=cv2.contourArea,reverse=True)

#the loop extracts the boundary contours of the page
for c in contours:
    p=cv2.arcLength(c,True)
    approx=cv2.approxPolyDP(c,0.02*p,True)

    if len(approx)==4:
        target=approx
        break
approx=mapp(target) #find endpoints of the sheet

pts=np.float32([[0,0],[1061,0],[1061,750],[0,750]])  #map to 800*800 target window

op=cv2.getPerspectiveTransform(approx,pts)  #get the top or bird eye view effect
dst=cv2.warpPerspective(orig,op,(1061,750))


cv2.imshow("Scanned",dst)
cv2.imwrite("result.jpg",dst)

# press q or Esc to close
cv2.waitKey(0)
cv2.destroyAllWindows()

    
        


