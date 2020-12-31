import cv2,time,argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("-lt","--line_thickness",help = "Line thickness in Scan Freeze filter",type = int,default = 2)
ap.add_argument("-v","--vid_input",help = "Video input path",type = str)
args = vars(ap.parse_args())

thickness = args["line_thickness"]

if args["vid_input"] != None:
    cap = cv2.VideoCapture(args["vid_input"])
else:
    cap = cv2.VideoCapture(0)
    
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

final_array = np.zeros((frame_height,frame_width,3),dtype = np.uint8)
x = 0

while(True):
    ret,frame = cap.read()
    if(ret == False):
        print("Camera not opened !!")
        break
    if(x >= thickness):
        final_array[:,x-thickness:x] = frame[:,x-thickness:x]
        final_array[:,x:] = frame[:,x:]
    else:
        final_array = frame
        
    cv2.line(final_array,(x,0),(x,frame.shape[0]),color = (0,255,0),thickness = thickness)
    cv2.imshow("Feed",final_array)
    ret_key = cv2.waitKey(1)
    
    if(x == frame.shape[1]):
        cv2.imwrite("ScanFreeze-Output.jpg",final_array[:,:x-thickness])
        break
    if(ret_key == 27):
        break
    x+=1

cap.release()
cv2.destroyAllWindows()