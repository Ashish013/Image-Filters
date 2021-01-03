import cv2,os,dlib,requests,argparse
import numpy as np
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("-vi","--video_input",help = "path to input_video")
args = vars(ap.parse_args())

def file_downloader(link,destination,chunk_size = 500):
  '''Helper function to download files using url link via the python requests module'''
  response = requests.get(link,stream = True)
      
  if (os.path.exists(destination) == False):
    with open(destination,"wb") as file:
      for chunk in response.iter_content(chunk_size): 
        if chunk:
          file.write(chunk)

def points2array(points):
  '''Helper function that converts the co-ordinates from points format of dlib to an array format'''
  landmarks = []
  for i in range(len(points)):
    landmarks.append([points[i].x,points[i].y])
  return landmarks

  
# Downloads dlib's landmark detection model and also the images to be overlaid on the face
print("Downloading files.....")
if(os.path.exists("shape_predictor_68_face_landmarks.dat") == False):
    file_downloader("https://github.com/JeffTrain/selfie/raw/master/shape_predictor_68_face_landmarks.dat","shape_predictor_68_face_landmarks.dat")
if(os.path.exists("specs.jpg") == False):
    file_downloader("https://ih1.redbubble.net/image.520905129.1812/flat,750x1000,075,f.u1.jpg","specs.jpg")
if(os.path.exists("cigar.jpg") == False):
    file_downloader("https://i.pinimg.com/originals/96/0b/af/960bafd7ea7aa70b93ebb654230100d0.png","cigar.jpg")
print("Completed !")
    
# Loads the spectacles image and crops it
specs = np.array(cv2.imread("specs.jpg"),dtype = np.uint8)
specs_crop = specs[420:498,179:567]
inv_specs_crop = cv2.bitwise_not(specs_crop)

# Loads the cigar image
cigar = np.array(cv2.imread("cigar.jpg"),dtype = np.uint8)

# Initializes the face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')

# Initializes the capture from the source
if args["video_input"] != None:
    cap = cv2.VideoCapture(args["video_input"])
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    writer = cv2.VideoWriter("output.avi",fourcc,cap.get(5),(int(cap.get(3)),int(cap.get(4))))
else:
    cap = cv2.VideoCapture(0)

while(cap.isOpened()):
    
    ret,frame = cap.read()
    
    if(ret == False):
        break
    
    if(np.all(frame) == None):
        break
    
    input_img = frame

    try:
        detections = detector(input_img,1)
    except:
        # In case no face are detected, live feed is shown
        cv2.imshow("Feed",input_img)
        continue
    
    for rect in detections:
      # Facial landmarks are identified and reformatted into an array type
      landmarks = predictor(input_img,rect)
      landmarks = points2array(landmarks.parts())

      # Calculating spectacles landmarks locations
      specs_left = landmarks[0]
      specs_right = landmarks[16]
      
      # Eye width is used to tweak the spectacle land mark positions calculated below.
      eyewidth = max(landmarks[40][1] - landmarks[38][1],landmarks[44][1] - landmarks[46][1])

      specs_leftup = [specs_left[0],int(specs_left[1] - 1.5 * eyewidth)]
      specs_rightup = [specs_right[0],int(specs_right[1] - 1.5 * eyewidth)]

      specs_leftdown = [specs_left[0],int(specs_left[1] + 1.5 * eyewidth)]
      specs_rightdown = [specs_right[0],int(specs_right[1] + 1.5 * eyewidth)]

      pts_src = np.array([(0,0),(specs_crop.shape[1]-1,0),(specs_crop.shape[1]-1,specs_crop.shape[0]-1),(0,specs_crop.shape[0]-1)])
      pts_dst = np.asarray([specs_leftup,specs_rightup,specs_rightdown,specs_leftdown])
      H = cv2.findHomography(pts_src,pts_dst,cv2.RANSAC)[0]
      
      # Spectacles are wrapped on to the input image
      specs_mask_inv = cv2.warpPerspective(inv_specs_crop,H,(input_img.shape[1],input_img.shape[0]))
      specs_mask = cv2.bitwise_not(specs_mask_inv)
      input_img = cv2.bitwise_and(specs_mask,input_img)
      
      #Calculating cigar landmark locations
      cigar_leftup = landmarks[62]
      cigar_leftdown = landmarks[57]
      cigar_rightup = [landmarks[13][0],cigar_leftup[1]]
      cigar_rightdown = [landmarks[13][0],cigar_leftdown[1]]

      pts_src = np.array([(0,0),(cigar.shape[1]-1,0),(cigar.shape[1]-1,cigar.shape[0]-1),(0,cigar.shape[0]-1)])
      pts_dst = np.array([cigar_leftup,cigar_rightup,cigar_rightdown,cigar_leftdown])
      H = cv2.findHomography(pts_src,pts_dst,cv2.RANSAC)[0]

      # Cigar is wrapped on to the input image
      cigar_out = cv2.warpPerspective(cigar,H,(input_img.shape[1],input_img.shape[0]))
      cigar_out_gray = cv2.cvtColor(cigar_out,cv2.COLOR_RGB2GRAY)
      cigar_mask_inv = cv2.threshold(cigar_out_gray,0,255,cv2.THRESH_BINARY)[1]
      cigar_mask = cv2.bitwise_not(cigar_mask_inv)
    
      # Final output image is calculated
      mask = cv2.bitwise_and(input_img,input_img,mask = cigar_mask)
      input_img = cv2.bitwise_or(mask,cigar_out)
    
    cv2.imshow("Feed",input_img)
    if args["video_input"] != None:
        writer.write(input_img)
        
    # Breaks the feed loop when ESC is pressed
    if(cv2.waitKey(1) == 27):
        break
   
    
cv2.imwrite("thug-life.jpg",input_img)

# Releases the resources allocated to the program 
# and destroys all the windows of the program
cap.release()
if args["video_input"] != None:
    writer.release()
cv2.destroyAllWindows()
