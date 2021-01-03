import cv2,time,argparse
import numpy as np
import matplotlib.pyplot as plt
from ssim_utils import generate_ssim_mask
from rcnn_utils import generate_rcnn_mask
from utils import files_downloader

ap = argparse.ArgumentParser(description = "Filter that captures instances, every '-time' seconds")
ap.add_argument("-m","--method",required = True, help = "Method used for detection",choices = ["ssim","rcnn"], type = str)
ap.add_argument("-v","--vid_input",help = "Video input path",type = str)
ap.add_argument("-t","--time",help = "Time difference between 2 instances", default = 5, type = int)
ap.add_argument("-f","--font", help = "Font to display timer", default = "cv2.FONT_HERSHEY_COMPLEX", type = str)
args = vars(ap.parse_args())

if args["vid_input"] != None:
    cap = cv2.VideoCapture(args["vid_input"])
else:
    cap = cv2.VideoCapture(0)

# Important args used in the script
method = args["method"]
buffer_time = args["time"]
font = eval(args["font"])
start_time = time.time()
# Offset position of the timer from the ends of the frame
offset = 75

# Triggers used to control the program flow
prev_num = 1
first_snap = False
first_frame = False

while(cap.isOpened()):
    ret,frame = cap.read()
    
    if ret == False:
        break
    if np.all(frame) == None:
        break
        
    if(first_snap == False):
        stitched_img = frame
        
    if(method == "ssim"):
        if(first_frame == False):
            print("Capturing Backgound.........Completed !")
            # Captures the static background in the first frame,
            # which is used later for computing ssim
            bg = frame
            first_frame = True
            
        thresh = generate_ssim_mask(frame,bg)
        inv_thresh = cv2.bitwise_not(thresh)
                
    elif(method == "rcnn"):
        if(first_frame == False):
            # Downloads the required files for applying
            # Mask-RCNN, in the first frame
            rcnn_file_path = files_downloader()
            first_frame = True
            
        thresh = generate_rcnn_mask(frame,rcnn_file_path = rcnn_file_path )
        inv_thresh = cv2.bitwise_not(thresh)
    
    fg_mask = cv2.bitwise_and(frame,frame,mask = thresh)
    bg_mask = cv2.bitwise_and(stitched_img,stitched_img,mask = inv_thresh)
    
    # The final image after masking is stored in temp which is copied to
    # stitched_img variable after every 'buffer_time' seconds 
    temp = cv2.bitwise_or(fg_mask,bg_mask)
    time_diff = int(time.time() - start_time)
    
    if((time_diff % buffer_time == 0) and time_diff >= prev_num):
        if(first_snap == False):
            first_snap = True
        stitched_img = temp.copy()
        cv2.putText(temp,"Snap !",(temp.shape[1] - offset - 100, offset),fontFace = font,fontScale = 1,color = (255,255,255),thickness = 2)
        prev_num = time_diff+1
    
    else:
        cv2.putText(temp,str(time_diff % buffer_time),(temp.shape[1] - offset, offset),fontFace = font,fontScale = 1.5,color = (255,255,255),thickness = 2)
    
    val = cv2.waitKey(1)
    cv2.imshow("Image",temp)
    
    #Breaks out of the loop when ESC key is pressed ! 
    if (val == 27):
        break

cv2.imwrite("TimeFreezeFilter.jpg",stitched_img)
cap.release()
cv2.destroyAllWindows()
