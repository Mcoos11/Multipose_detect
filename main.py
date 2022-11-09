import cv2
import random, os, re
import numpy as np
import mediapipe as mp
import operator as op

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# global variables
pose_estimator = []

cap = cv2.VideoCapture(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_inputs', 'input1.mp4'))
# cap = cv2.VideoCapture(3)

whT = 320 # word hight Target
confThreshold = .5
nmsThreshold = .3

classesFile = 'coco.names'
classNames = []
with open(classesFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

class person:
    x = None
    y = None
    width = None
    height = None
    id = None
    lines_color = None
    points_color = None

    pose_estimator = "empty"

    def __init__(self, x, y, w, h, id):
        self.x = x
        self.y = y
        self.width = w
        self.height = h
        self.id = id
        self.lines_color = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
        self.points_color = (random.randint(0,255), random.randint(0,255), random.randint(0,255))

global objects
objects = []
global CamH, CamW, CamC, CamDiag

modelConfiguration = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'yolo', 'yolov3-608.cfg')
modelNames = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'yolo', 'yolov3-608.weights')
net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelNames)

cv_info = str(cv2.getBuildInformation().strip().split('\n')[99])
cv_info = cv_info.replace(re.search(".+:\s+", cv_info).group(0),"")[:3]

if cv_info == 'YES':
    print('USING CUDA')
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
else:
    print('USING CPU')
    modelConfiguration = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'yolo', 'yolov3-tiny.cfg')
    modelNames = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'yolo', 'yolov3-tiny.weights')

    net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelNames)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

def setMinDistance():
    global min_distance
    if CamDiag <= 1469:
        min_distance = CamDiag * .08
    elif CamDiag > 1469 and CamDiag <= 2210:
        min_distance = CamDiag * .19
    else:
        print("CAM RESOLUTION DON'T SUPPORTED")
        min_distance = 1000

    print("distance < ", min_distance)
def getId(x, y, w, h):
    distances = []
    id = None
    if not objects:
        objects.append(person(x, y, w, h, 0))
    else:
        for object in objects:
            distances.append(((object.x - x)**2 + (object.y - y)**2)**.5)
        distance = min(distances)

        if(distance < min_distance): #przemyśleć
            id = distances.index(distance)
            objects[id].id = id
            objects[id].x = x
            objects[id].y = y
            objects[id].width = w
            objects[id].hight = h
        else:
            id = objects.index(op.itemgetter(-1)(objects)) + 1
            objects.append(person(x, y, w, h, id))
    return id

def findObjects(datasets, image):
    hT, wT, cT = image.shape
    bBox = []
    classIds = []
    confs = []

    for dataset in datasets:
        for line in dataset:
            scores = line[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]

            if confidence > confThreshold:
                w, h = int(line[2] * wT), int(line[3] * hT)
                x, y = int(line[0] * wT - w/2), int(line[1] * hT - h/2) # centrum

                bBox.append([x, y, w, h])
                classIds.append(classId)
                confs.append(float(confidence))
        break
    indices = cv2.dnn.NMSBoxes(bBox, confs, confThreshold, nmsThreshold) # usuwa zduplikowane wyktycia
    for i in indices:
        if classNames[classIds[i]] == 'person':
            box = bBox[i]
            x, y, w, h = box[0], box[1], box[2], box[3]
            x = x - int(CamDiag * .015)
            w = w + int(CamDiag * .015)
            id = getId(x, y, w, h)
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 255))
            cv2.putText(image, f'{classNames[classIds[i]].upper()} - {int(confs[i] * 100)}% : {id}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)

success, img = cap.read()
CamH, CamW, CamC = img.shape
CamDiag = (CamH**2 + CamW**2)**.5
setMinDistance()

while cap.isOpened():
    success, img = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue
    ##YOLO------------------

    # sieć potrzebuje formatu blob, więc konversja
    blob = cv2.dnn.blobFromImage(img, 1/255, (whT, whT), [0, 0, 0], 1, crop=False)
    net.setInput(blob)

    layerNames = net.getLayerNames()
    outputNames = [layerNames[i - 1] for i in net.getUnconnectedOutLayers()]

    outputs = net.forward(outputNames)
    findObjects(outputs, img)

    ##MEDIAPIPE-------------

    for object in objects:
        if object.pose_estimator == "empty":
            object.pose_estimator = mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6)

        if object.x < 0:
            object.x = 0
        if object.y < 0:
            object.y =0

        masked_img = img[object.y:object.y+object.height, object.x:object.x+object.width]
        if not masked_img.any():
            print(object.y,object.height, object.x,object.width)
            continue
        masked_img = cv2.cvtColor(masked_img, cv2.COLOR_BGR2RGB)
        masked_img.flags.writeable = False

        results = object.pose_estimator.process(masked_img)

        masked_img.flags.writeable = True
        masked_img = cv2.cvtColor(masked_img, cv2.COLOR_RGB2BGR)

        mp_drawing.draw_landmarks(masked_img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=object.points_color, thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=object.lines_color, thickness=2, circle_radius=2))

        img[object.y:object.y + object.height, object.x:object.x + object.width] = masked_img

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()