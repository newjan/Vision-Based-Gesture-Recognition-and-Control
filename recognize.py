

from keras.models import model_from_json 
from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img
import os
import numpy as np
import shutil  
import json
import sys
import cv2 
import asyncio
import time
from VideoGet import VideoGet
from VideoShow import VideoShow
import pyautogui

sys.path.append('../lib/')
import time 

labels_want = ['Swiping Left',
              'Swiping Right',
              'Swiping Down',
              'Swiping Up',
              'Rolling Hand Forward',
              'Rolling Hand Backward',
               'Turning Hand Clockwise',
               'Turning Hand Counterclockwise',
               'Zooming In With Full Hand',
               'Zooming Out With Full Hand',
               'Shaking Hand',
               'Drumming Fingers',
               'No gesture'
              ]



image_size=(50,88,3)

class GestureBrain():

    def __init__(self,  model ):
        self.curr_stream = []
        self.model = model 
        self.counter  = 0 

    def adjust_judge_sequence(self):
        frame_diff = len(self.curr_stream) - 40
        print("diff is", frame_diff)
        if  frame_diff == 0 :
            return self.curr_stream
        elif frame_diff > 0 :
            return self.curr_stream[frame_diff:]
        else :
            return self.curr_stream[:1] * abs(frame_diff) + self.curr_stream

    def prepreocess_img (self,  img_array):
        return (img_array / 255. )

    def regonize(self):

        x = []         
        x.append(self.curr_stream)

        X = np.array(x)
        result = self.model.predict(X)
        index = self.model.predict_classes(X)
        print(result)
        #self.curr_stream = []
        if index != 13 :
           print("cleaning!")
           self.curr_stream = []

        return index 

    def img_num (self ):
        return len( self.curr_stream)

    # build a first in first out queue for img
    # and keep 40 frames length 
    def push_img (self,  img ):
        img = self.prepreocess_img(img )
        self.curr_stream.append(img)

        if len( self.curr_stream ) > 40 :
            self.curr_stream.pop(0)

def adjust_sequence_length( frame_files):
    """Adjusts a list of files pointing to video frames to shorten/lengthen
    them to the wanted sequence length (self.seq_length)"""
    frame_diff = len(frame_files) - 40 
    if frame_diff == 0:
        # No adjusting needed
        return frame_files
    elif frame_diff > 0:
        # Cuts off first few frames to shorten the video
        return frame_files[frame_diff:]
    else:
        # Repeats the first frame to lengthen video
        return frame_files[:1] * abs(frame_diff) + frame_files

def preprocess_image(image_array):
        return (image_array / 255. )

def build_sequence( ):
    path = "./TestImgv3/"
    frame_files = os.listdir(path)
    # add sorted, so we can recognize the currect sequence
    frame_files = sorted(frame_files)
    print(frame_files)
    sequence = []

    # Adjust length of sequence to match 'self.seq_length'
    frame_files = adjust_sequence_length(frame_files)

    frame_paths = [os.path.join(path, f) for f in frame_files]
    for frame_path in frame_paths:
        image = load_img(frame_path, target_size=image_size)
        image_array = img_to_array(image)
        image_array = preprocess_image(image_array)

        sequence.append(image_array)

    return np.array(sequence)

def check_value_pics( model):
    x = [] 
    sequence = build_sequence()
    x.append(sequence)

    X = np.array(x)
    #print("---- val --- ",model.predict(X))
    index = model.predict_classes(X)

    print("----- we guess is -----",labels_want[index[0]])

    return index
def vol_up(hold_time):
    start = time.time()
    while time.time() - start < hold_time:
        pyautogui.hotkey('ctrl','up')
    
def vol_down(hold_time):
    start = time.time()
    while time.time() - start < hold_time:
        pyautogui.hotkey('ctrl','down')
        
def video_seekf(hold_time):
    start = time.time()
    while time.time() - start < hold_time:
        pyautogui.press('right')

def video_seekb(hold_time):
    start = time.time()
    while time.time() - start < hold_time:
        pyautogui.press('left')
    
# version 2 
def gesture_guy():
    model = load_model('my_model.h5')
    model.load_weights('model_weights.h5')    

    gb = GestureBrain(model)
    print("start ======")
    counter = 0
    i=0
    cap = cv2.VideoCapture(0)
    (grabbed, frame) = cap.read()
    video_shower = VideoShow(frame).start()
    mode_i = None
    action2 = None
    while True :
        (grabbed, frame) = cap.read()
        if not grabbed or video_shower.stopped:
            video_shower.stop()
            break
        cv2.putText(frame, "Current Gesture: {}".format(action2),(10, 450), cv2.FONT_HERSHEY_TRIPLEX,0.8, (0, 255, 255))
        cv2.putText(frame, "Mode: {}".format(mode_i),(10, 400), cv2.FONT_HERSHEY_TRIPLEX,0.8, (0, 255, 255))
        video_shower.frame = frame
        counter += 1
        screen = np.array(cv2.resize(frame,(88,50)))
        
        gb.push_img(screen)
        if counter != 40 :
            continue 
        else:
            counter = 0
        
        mode = ['Recognition mode','Explorer mode','Photo mode','Video mode']
        selectMode = mode[i]
        print(selectMode)
        action = gb.regonize()
        print("predict type :", action )
        if selectMode == mode[0]:
            if action == 0 :
                print(labels_want[0])
            elif action == 1 :
                print(labels_want[1])
            elif action == 2 :
                print(labels_want[2])
            elif action == 3 :
                print(labels_want[3])
            elif action == 4 :
                print(labels_want[4])            
            elif action == 5 :
                print(labels_want[5])
            elif action == 6 :
                print(labels_want[6])            
            elif action == 7 :
                print(labels_want[7])
            elif action == 8 :
                print(labels_want[8])
            elif action == 9 :
                print(labels_want[9])

            elif action == 10 :
                print(labels_want[10])
                i += 1
                if i != 4 :
                    continue 
                else:
                    i = 0

            elif action == 11 :
                print(labels_want[11])
            elif action == 12 :
                print(labels_want[12])
        if selectMode == mode[2]:
            if action == 0 :
                print(labels_want[0])
                pyautogui.press('left')
            elif action == 1 :
                print(labels_want[1])
                pyautogui.press('right')
            elif action == 4 :
                print(labels_want[4])
                pyautogui.press('left')
            elif action == 5 :
                print(labels_want[5])
                pyautogui.press('right')
            elif action == 6 :
                print(labels_want[6])
#                 pyautogui.hotkey('altleft','f4')
            elif action == 7:
                print(labels_want[7])
                pyautogui.hotkey('enter')
            elif action == 8:
                print(labels_want[8])
                pyautogui.hotkey('ctrl','+')
            elif action == 9 :
                print(labels_want[9])
                pyautogui.hotkey('ctrl','-')

            elif action == 10:
                print(labels_want[10])
                i += 1
                if i != 4 :
                    continue 
                else:
                    i = 0

            elif action == 11:
                print(labels_want[11])
                pyautogui.press('winleft')

        if selectMode == mode[3]:
            if action == 0 :
                print(labels_want[0])
                video_seekb(0.4)    
            elif action == 1 :
                print(labels_want[1])
                video_seekf(0.4)                                  
            elif action == 4 :
                print(labels_want[4])
                vol_up(0.8)
            elif action == 5 :
                print(labels_want[5])
                vol_down(0.8)
            elif action == 6 :
                print(labels_want[6])
                pyautogui.press('space')
            elif action == 7 :
                print(labels_want[6])
                pyautogui.press('space')            
            elif action == 9 :
                print(labels_want[9])

            elif action == 10:
                print(labels_want[10])
                i += 1
                if i != 4 :
                    continue 
                else:
                    i = 0                

            elif action == 11:
                print(labels_want[11])
                
        if selectMode == mode[1]:
            if action == 0 :
                print(labels_want[0])
                pyautogui.press('left')
            elif action == 1 :
                print(labels_want[1])
                pyautogui.press('right')                                  
            elif action == 2 :
                print(labels_want[2])
                pyautogui.press('down')
            elif action == 3 :
                print(labels_want[3])
                pyautogui.press('up')                
            elif action == 4 :
                print(labels_want[4])
                pyautogui.press('enter')
            elif action == 5 :
                print(labels_want[5])
                pyautogui.press('backspace')            
            elif action == 6 :
                print(labels_want[6])
#                 pyautogui.hotkey('altleft','f4')
            elif action == 9 :
                print(labels_want[9])

            elif action == 10:
                print(labels_want[10])
                i += 1
                if i != 4 :
                    continue 
                else:
                    i = 0

            elif action == 11:
                print(labels_want[11])
                pyautogui.press('winleft')   
#         if selectMode == mode[4]:
#             if action == 0 :
#                 print(labels_want[0])
#                 pyautogui.press('left')
#             elif action == 1 :
#                 print(labels_want[1])
#                 pyautogui.press('right')                                  
#             elif action == 2 :
#                 print(labels_want[2])
#                 pyautogui.press('down')
#             elif action == 3 :
#                 print(labels_want[3])
#                 pyautogui.press('up')
#             elif action == 4 :
#                 print(labels_want[4])
#                 pyautogui.press('pagedown')
#             elif action == 5 :
#                 print(labels_want[5])
#                 pyautogui.press('pageup')


#             elif action == 10:
#                 print(labels_want[10])
#                 i += 1
#                 if i != 5 :
#                     continue 
#                 else:
#                     i = 0

#             elif action == 11:
#                 print(labels_want[10])
#                 pyautogui.press('winleft')
            
#         cv2.putText(frame, "{}".format(action),(10, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255))
        mode_i = mode[i]
        action2=labels_want[action[0]]
                
        

def seeing_guy():
    model = load_model('my_model.h5')
    model.load_weights('model_weights.h5')    

    index = 0 
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 10)
    cv2.namedWindow('Capture')


    #os.mkdir('./TestImgv3')    
    while True:
        ret, frame = cap.read()

        cv2.imshow('Capture',frame)
        
        
        screen  = frame

        name = str(index).zfill(5)
        file_name =  name +".jpg"

        cv2.imwrite('./TestImgv3/'+file_name, screen)

        index += 1
        print("get pic ",index)

       

        if index == 41 :
            action =  check_value_pics( model)

            if action == 0 :
                print(labels_want[0])
            elif action == 1 :
                print(labels_want[1])
            elif action == 2 :
                print(labels_want[2])
            elif action == 3 :
                print(labels_want[3])
            elif action == 4 :
                print(labels_want[4])
            elif action == 5 :
                print(labels_want[5])
            elif action == 6 :
                print(labels_want[6])
            elif action == 7 :
                print(labels_want[7])
            elif action == 8 :
                print(labels_want[8])
            
            
            index = 0 
            input()
            shutil.rmtree('./TestImgv3')
            os.mkdir('./TestImgv3')
            cv2.waitKey(20)

            if cv2.waitKey(20) & 0xFF == ord('q'):
                        break;  
    cap.release()
    cv2.destroyAllWindows()  



if __name__ == '__main__':
     gesture_guy()
   
