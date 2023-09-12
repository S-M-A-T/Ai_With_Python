#importing libraries
import cv2
import numpy as np
import pyttsx3

#cv2.dnn.readNet to use a neural network model for text detection in an image
net = cv2.dnn.readNet('yolov3 (1).weights', 'yolov3.cfg')
model = cv2.dnn_DetectionModel(net)
#to set preprocessing parameters for frame
model.setInputParams(size=(416, 416), scale=1/255)

#load class list or coco.names file
#save all the names in file o the list classes
classes = []
with open("coco.names", "r") as f:
    for class_name in f.readlines():
        class_name = class_name.strip()
        classes.append(class_name)
print("Object list")
print(classes)
#get layers of the network
layer_names = net.getLayerNames()
#Determine the output layer names from the YOLO model
output_layers = [layer_names[i-1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

#initializing camera
cap = cv2.VideoCapture(0)
#to set the standard resolution of the camera
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    #to get frames
    ret,frame =cap.read()
    height, width, _ = frame.shape

    # using blob function of opencv to preprocess image
    blob = cv2.dnn.blobFromImage(frame, 1/255, (416, 416), (0,0,0), swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(output_layers)
    # Showing Information on the screen
    boxes = []
    confidences = []
    class_ids = []
    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.2:
                center_x = int(detection[0]*width)
                center_y = int(detection[1]*height)
                w = int(detection[2]*width)
                h = int(detection[3]*height)

                x = int(center_x - w/2)
                y = int(center_y - h/2)

                boxes.append([x, y, w, h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.4)
    # ensure at least one detection exists
    if len(indexes)>0:
        # loop over the indexes we are keeping
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = str(round(confidences[i],2))
            color = colors[i]
            #to draw rectangle on frame
            cv2.rectangle(frame, (x,y), (x+w, y+h), color, 2)
            #to put text on frame
            cv2.putText(frame, label + " " + confidence, (x, y+20), cv2.FONT_HERSHEY_PLAIN, 2, (255,255,255), 2)
            #the instance of the initialized pyttsx3 package is stored in the engine variable.
            engine = pyttsx3.init()
            #say() function in the pyttsx3 package that takes a string value and speaks it out
            engine.say(label)
            engine.runAndWait()
#to show frames
    cv2.imshow('Image', frame)
    prev_frame_labels = [str(classes[class_ids[i]]) for i in indexes.flatten()]
    #to put the frame on hold
    key = cv2.waitKey(1)
    if key==27:
        break
#to release the camera
cap.release()
#to close all windows after exiting
cv2.destroyAllWindows()