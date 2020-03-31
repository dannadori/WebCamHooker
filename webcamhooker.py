import cv2
from subprocess import Popen, PIPE
from PIL import Image
import numpy as np
import argparse
from enum import IntEnum, auto

'''
Command line arguments
'''
def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description='specify input and output device')
parser.add_argument('--input_video_num', type=int, required=True,
                    help='input video device number. ex) if input is /dev/video0 then the value is 0')
parser.add_argument('--output_video_dev', type=str, required=True,
                    help='input video device. ex) /dev/video2')
parser.add_argument('--emotion_mode', type=str2bool, required=False, default=False,
                    help='enable emotion mode')


'''
Mode definition
'''
class modes(IntEnum):
    SIMPLE_SMILE_MODE = auto()
    
    
'''
Classifiers
'''
face_classifier_classifier = None
smile_cascade_classifier   = None

'''
Images
'''
smile_mark_img = None

'''
Path for resources
'''
face_cascade_path  = './models/haarcascade_frontalface_default.xml'
smile_cascade_path = './models/haarcascade_smile.xml'
emotion_model_path = './models/fer2013_mini_XCEPTION.102-0.66.hdf5'
gender_model_path  = './models/gender_models/simple_CNN.81-0.96.hdf5'

smile_img_path     = './images/mark_face_laugh.png'


def load_resources(mode):
    global face_classifier_classifier
    face_classifier_classifier = cv2.CascadeClassifier(face_cascade_path)
    print('LOAD!!!!!!',mode)

    if mode == modes.SIMPLE_SMILE_MODE :
        global smile_cascade_classifier, smile_mark_img
        print('LOAD!!!!!!')
        smile_cascade_classifier = cv2.CascadeClassifier(smile_cascade_path)
        smile_mark_img           = cv2.imread(smile_img_path)
    

def paste(img, imgback, x, y, angle, scale):  
    r   = img.shape[0]
    c   = img.shape[1]
    rb  = imgback.shape[0]
    cb  = imgback.shape[1]
    hrb = round(rb/2)
    hcb = round(cb/2)
    hr  = round(r/2)
    hc  = round(c/2)

    # Copy the forward image and move to the center of the background image
    imgrot = np.zeros((rb,cb,3),np.uint8)
    imgrot[hrb-hr:hrb+hr,hcb-hc:hcb+hc,:] = img[:hr*2,:hc*2,:]

    # Rotation and scaling
    M = cv2.getRotationMatrix2D((hcb,hrb),angle,scale)
    imgrot = cv2.warpAffine(imgrot,M,(cb,rb))
    # Translation
    M = np.float32([[1,0,x],[0,1,y]])
    imgrot = cv2.warpAffine(imgrot,M,(cb,rb))

    # Makeing mask
    imggray = cv2.cvtColor(imgrot,cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(imggray, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    # Now black-out the area of the forward image in the background image
    img1_bg = cv2.bitwise_and(imgback,imgback,mask = mask_inv)

    # Take only region of the forward image.
    img2_fg = cv2.bitwise_and(imgrot,imgrot,mask = mask)

    # Paste the forward image on the background image
    imgpaste = cv2.add(img1_bg,img2_fg)

    return imgpaste


def edit_frame(frame):
    
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier_classifier.detectMultiScale(gray, 1.1, 5)
    
    for (x,y,w,h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        if mode == modes.SIMPLE_SMILE_MODE:
            smiles = smile_cascade_classifier.detectMultiScale(roi_gray, scaleFactor= 1.2, minNeighbors=20, minSize=(100, 100))
            if len(smiles) > 0 :
                frame = paste(smile_mark_img, frame, 0, 0, 20, 2.0)
                
    return frame


if __name__=="__main__":
    args             = parser.parse_args()
    print(args)
    input  = args.input_video_num
    output = args.output_video_dev
    cap    = cv2.VideoCapture(input)
    if args.emotion_mode == True:
        pass
    else:
        mode = modes.SIMPLE_SMILE_MODE

    print(mode)
    load_resources(mode)

    print('web camera hook start!')
    p = Popen(['ffmpeg', '-y', '-i', '-', '-pix_fmt', 'yuyv422', '-f', 'v4l2', output], stdin=PIPE)


    try:
        while True:
            ret,im = cap.read()
            im     = edit_frame(im)
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            im     = Image.fromarray(np.uint8(im))
            im.save(p.stdin, 'JPEG')
    except KeyboardInterrupt:        
        pass
    
    p.stdin.close()
    p.wait()

    print('web camera hook fin!')
    
