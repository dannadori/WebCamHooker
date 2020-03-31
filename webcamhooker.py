import cv2
from subprocess import Popen, PIPE
from PIL import Image
import numpy as np
import argparse
from enum import IntEnum, auto
import sys, math, os

from keras.models import load_model


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
    EMOTION_MODE      = auto()    
'''
Classifiers
'''
face_classifier_classifier = None
smile_cascade_classifier   = None
emotion_classifier         = None
gender_classifier          = None

'''
Labels
'''
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
gender_labels  = ['woman', 'man']

'''
Images
'''
smile_mark_image = None
emotion_images   = None
'''
Path for resources
'''
face_cascade_path  = './models/haarcascade_frontalface_default.xml'
smile_cascade_path = './models/haarcascade_smile.xml'
emotion_model_path = './models/fer2013_mini_XCEPTION.110-0.65.hdf5'
gender_model_path  = './models/simple_CNN.81-0.96.hdf5'

smile_img_path     = './images/mark_face_laugh.png'

emtion_img_paths   = [
    [
        'images/pose_sugoi_okoru_woman.png',
        'images/business_woman2_4_think.png',
        'images/business_woman2_2_shock.png',
        'images/business_woman1_4_laugh.png',
        'images/business_woman1_3_cry.png',
        'images/business_woman2_3_surprise.png',
        None,
    ],
    
    [
        'images/pose_sugoi_okoru_man.png',
        'images/business_man2_4_think.png',
        'images/business_man2_2_shock.png',
        'images/business_man1_4_laugh.png',
        'images/business_man1_3_cry.png',
        'images/business_man2_3_surprise.png',
        None,
    ]    
]


def load_resources(mode):
    global face_classifier_classifier
    face_classifier_classifier = cv2.CascadeClassifier(face_cascade_path)

    if mode == modes.SIMPLE_SMILE_MODE :
        global smile_cascade_classifier, smile_mark_image
        print('model loding for mode: SIMPLE_SMILE_MODE', mode)
        smile_cascade_classifier = cv2.CascadeClassifier(smile_cascade_path)
        smile_mark_image           = cv2.imread(smile_img_path)
    if mode == modes.EMOTION_MODE:
        global emotion_classifier, gender_classifier, emotion_images
        print('model loding for mode: EMOTION_MODE', mode)
        emotion_classifier = load_model(emotion_model_path, compile=False)
        gender_classifier  = load_model(gender_model_path,  compile=False)
        images = []
        for path_per_gender in emtion_img_paths:
            images_per_gender = []
            for path in path_per_gender:
                #if os.path.exists(path) == False:
                #    print(sys.stderr, f'----------------------- {path} is not foound')
                images_per_gender.append(cv2.imread(path))
                
            images.append(images_per_gender)
        emotion_images = images
        

def paste(img, imgback, x, y, angle, scale):
    r   = img.shape[0]
    c   = img.shape[1]
    rb  = imgback.shape[0]
    cb  = imgback.shape[1]    
    #print(sys.stderr, f'(1) -> {r}, {c},    {rb},{cb}')
    
    if img.shape [0] > imgback.shape[0] or img.shaoe[1] > imgback.shape[1]:
        h_ratio = imgback.shape[0] / img.shape[0]
        w_ratio = imgback.shape[1] / img.shape[1]
        if h_ratio < w_ratio:
            new_h = int(img.shape[0] * h_ratio)
            new_w = int(img.shape[1] * h_ratio)
        else:
            new_h = int(img.shape[0] * w_ratio)
            new_w = int(img.shape[1] * w_ratio)
        if new_h % 2 != 0:
            new_h += 1
        if new_w % 2 != 0:
            new_w += 1
            
        img = cv2.resize(img, (new_w, new_h))
        #print(sys.stderr, f'{new_h}, {new_w}')    
    r   = img.shape[0]
    c   = img.shape[1]
    rb  = imgback.shape[0]
    cb  = imgback.shape[1]    
    hrb = round(rb/2)
    hcb = round(cb/2)
    hr  = round(r/2)
    hc  = round(c/2)

    #print(sys.stderr, f'(2) -> {r}, {c},    {rb},{cb}')    
    
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

    return imgpaste, new_h, new_w

def apply_offsets(face_location, offsets):
    x, y, width, height = face_location
    x_off, y_off = offsets
    return (x - x_off, x + width + x_off, y - y_off, y + height + y_off)

def preprocess_input(x, v2=True):
    x = x.astype('float32')
    x = x / 255.0
    if v2:
        x = x - 0.5
        x = x * 2.0
    return x

def edit_frame(frame):
    
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier_classifier.detectMultiScale(gray, 1.1, 5)
    
    for (x,y,w,h) in faces:
        #cv2.rectangle(frame,(x,y),(x+w,y+h),(255, 0, 0),2)
        if mode == modes.SIMPLE_SMILE_MODE:
            roi_gray = gray[y:y+h, x:x+w]
            smiles = smile_cascade_classifier.detectMultiScale(roi_gray, scaleFactor= 1.2, minNeighbors=20, minSize=(100, 100))
            if len(smiles) > 0 :
                frame = paste(smile_mark_image, frame, 0, 0, 20, 2.0)
        if mode == modes.EMOTION_MODE:
            try:
                '''
                judge gender
                '''
                gender_offsets    = (10, 10)
                x1, x2, y1, y2    = apply_offsets((x,y,w,h), gender_offsets)
                gender_rgb        = frame[y1:y2, x1:x2]
                target_size       = gender_classifier.input_shape[1:3]
                gender_rgb        = cv2.resize(gender_rgb, target_size)
                
                gender_rgb        = preprocess_input(gender_rgb, False)
                gender_rgb        = np.expand_dims(gender_rgb, 0)
                gender_pred       = gender_classifier.predict(gender_rgb)
                gender_label_arg  = np.argmax(gender_pred[0])
                gender_score      = round(gender_pred[0][gender_label_arg]*100,2)
                gender_label      = gender_labels[gender_label_arg]
                
                '''
                gedge emotion
                '''
                emotion_offsets   = (0, 0)
                x1, x2, y1, y2    = apply_offsets((x,y,w,h), emotion_offsets)
                emotion_gray      = gray[y1:y2, x1:x2]
                target_size       = emotion_classifier.input_shape[1:3]
                emotion_gray      = cv2.resize(emotion_gray, target_size)
                
                emotion_gray      = preprocess_input(emotion_gray, True)
                emotion_gray      = np.expand_dims(emotion_gray, 0)
                emotion_gray      = np.expand_dims(emotion_gray, -1)
                emotion_pred      = emotion_classifier.predict(emotion_gray)
                emotion_label_arg = np.argmax(emotion_pred[0])
                emotion_score     = round(emotion_pred[0][emotion_label_arg]*100,2)
                emotion_label     = emotion_labels[emotion_label_arg]
                
                '''
                show result
                '''
                emotion_image     = emotion_images[gender_label_arg][emotion_label_arg]
                font_scale=1
                color = (40,40,40)
                thickness=2
                cv2.putText(frame, f'{gender_label}({gender_score}), {emotion_label}({emotion_score})',
                            (10,50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            font_scale, color, thickness, cv2.LINE_AA
                )

                if emotion_image is not None:
                    frame,h,w = paste(emotion_image, frame, +180, -100, 350, 0.5)
                else:
                    pass
                    #cv2.putText(frame, f'NONE!',
                    #            (100,200),
                    #            cv2.FONT_HERSHEY_SIMPLEX,
                    #            font_scale, color, thickness, cv2.LINE_AA
                    #)
                    
            except Exception as e:
                print(sys.stderr, e)
                #print(e)
                #raise e
                pass
                
    return frame


if __name__=="__main__":
    args             = parser.parse_args()
    print(args)
    input  = args.input_video_num
    output = args.output_video_dev
    cap    = cv2.VideoCapture(input)
    if args.emotion_mode == True:
        mode = modes.EMOTION_MODE
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
    
