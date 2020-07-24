from picamera.array import PiRGBArray
from picamera import PiCamera
import cv2
import numpy as np
import face_recognition
import pickle
import time

import firebase_admin
from firebase_admin import credentials
from firebase_admin import db

import dropbox
import os

dropbox_token = os.getenv('DROPBOX')
dbx = dropbox.Dropbox(dropbox_token)


face_cascade_name = './haarcascades/haarcascade_frontalface_alt.xml'
face_cascade = cv2.CascadeClassifier()
#-- 1. Load the cascades
if not face_cascade.load(cv2.samples.findFile(face_cascade_name)):
    print('--(!)Error loading face cascade')
    exit(0)


firebase_token = os.getenv('FIREBASE')
# Fetch the service account key JSON file contents
cred = credentials.Certificate(firebase_token)
dropurl = os.getenv('DROPURL')
# Initialize the app with a service account, granting admin privileges
firebase_admin.initialize_app(cred, {
    'databaseURL': dropurl
})

frame_count = 0
frame_interval = 8

frame_width = 640
frame_height = 480
frame_resolution = [frame_width, frame_height]
frame_rate = 16

# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.resolution = frame_resolution
camera.framerate = frame_rate
rawCapture = PiRGBArray(camera, size=(frame_resolution))
# allow the camera to warmup
time.sleep(0.1)

catured_image = './image/captured_image.jpg'
#catured = open(catured_image,'wb')

cnt = 0
# capture frames from the camera
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    start_time = time.time()
    # grab the raw NumPy array representing the image
    image = frame.array
    # store temporary catured image
    camera.capture(catured_image)
    # transform gray image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #-- Detect faces
    faces = face_cascade.detectMultiScale(gray)

    rois = [(y, x + w, y + h, x) for (x, y, w, h) in faces]

    encodings = face_recognition.face_encodings(rgb, rois)


    data = []
    visit = []
    true = 0
    false = 0
    is_visit = false
    # loop over the facial embeddings
    for encoding in encodings:
        # attempt to match each face in the input image to our known
        # encodings
        if(cnt == 0) :
            visit.append(encoding)

        if(cnt != 0):
            matches = face_recognition.compare_faces(data["encodings"],
                encoding)
            visit.append(encoding)
            if True in matches:
                true += 1
            if False in matches:
                false += 1

   
    if(len(visit) > 0) :
        data = {"encodings" : visit} 
        cnt += 1
        if(true >= false) : 
            is_visit = True
        if(false > true):
            is_visit = False
        
    
        if(is_visit == False):
            # Send Notice
            print("Send Notice")
            current = str(time.time())
            path = '/' + current[:10] + '.jpg'
            print(path)
            #cv2.imwrite(path, image)
            
            ref = db.reference('surveillance')
            box_ref = ref.child("visit")
            box_ref.update({
                'name': cnt,
                'time': time.time(),
                'path': path
            })
            dbx.files_upload(open(catured_image, "rb").read(), path)
            print(dbx.files_get_metadata(path))
            #os.remove(catured_image)

 
                
    end_time = time.time()
    process_time = end_time - start_time
    print("=== A frame took {:.3f} seconds".format(process_time))
    # show the output image
    cv2.imshow("Recognition", image)

    key = cv2.waitKey(1) & 0xFF
    # clear the stream in preparation for the next frame
    rawCapture.truncate(0)
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
