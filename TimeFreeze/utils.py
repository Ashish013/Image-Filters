import os,requests,cv2,shutil

def files_downloader(filename = "rcnn_files"):
    
    print("Downloading files for Mask-Rcnn.....")
    file_url = "http://download.tensorflow.org/models/object_detection/mask_rcnn_inception_v2_coco_2018_01_28.tar.gz"
    response = requests.get(file_url,stream = True)
    
    if (os.path.exists("file.tar.gz") == False and os.path.exists(f"{filename}") == False):
        with open("file.tar.gz","wb") as file:
            for chunk in response.iter_content(chunk_size = 1024): 
                if chunk:
                    file.write(chunk)
                    
        shutil.unpack_archive(os.getcwd() + "\\file.tar.gz")
        os.rename("mask_rcnn_inception_v2_coco_2018_01_28",f"{filename}")
        os.remove('file.tar.gz')

    text_file_url = "https://raw.githubusercontent.com/amikelive/coco-labels/master/coco-labels-2014_2017.txt"
    response = requests.get(text_file_url,stream = True)
    if(os.path.exists(f"{filename}/mscoco_labels_names.txt") == False):
        with open(f"{filename}/mscoco_labels_names.txt","wb") as file:
            for chunk in response.iter_content(chunk_size = 128): 
                if chunk:
                    file.write(chunk)

    file_url = "https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt"
    response = requests.get(file_url)
    if(os.path.exists(f"{filename}/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt") == False):
        with open(f"{filename}/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt","wb") as file:
            for chunk in response.iter_content(chunk_size = 10): 
                if chunk:
                    file.write(chunk)
    print("Download Completed !")
    return filename

def bgr2rgb(frame):
    return cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

def bgr2gray(frame):
    return cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)