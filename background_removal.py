#importing the library
import cv2
import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import os

# Load the background images
list_dir=os.listdir("Background")

img_list=[]

for img in list_dir:
    img=cv2.imread(f"Background/{img}")
    #Resize the image same as camera frame size, otherwise not working.
    down_width = 640
    down_height = 360
    down_points = (down_width, down_height)
    bg_img = cv2.resize(img,down_points)
    img_list.append(bg_img)

cap = cv2.VideoCapture(0)
#set the height and width of the camera
cap.set(3, 640)
cap.set(4, 360)

#initialize the segmentor model
segmentor = SelfiSegmentation()

img_index=0

while True:
    success, img = cap.read()
    
    img_seg=segmentor.removeBG(img,img_list[img_index],threshold=0.8)
    img_stack= cvzone.stackImages([img,img_seg],2, 1)
    cv2.imshow("Image", img_stack)
    key = cv2.waitKey(1) 
    
    # set key "a" for previous background
    # set key "d" for next background
    # set key "q" for exit
    if key== ord("a"):
        if img_index > 0:
            img_index -=1
        
    elif key== ord("d"):
        if img_index < len(img_list)-1:
            img_index +=1
        
    elif key== ord("q"):
        break   
  
# After the loop release the cap object
cap.release()
# Destroy all the windows
cv2.destroyAllWindows()