#from tflite_runtime.interpreter import Interpreter
import numpy as np
import cv2
import tensorflow as tf
from matplotlib import pyplot as plt

interpreter = tf.lite.Interpreter(model_path='model.tflite')
interpreter.allocate_tensors()
input_size = 192

global fallstatus
fallstatus = 0

# Dictionary to map joints of body part
KEYPOINT_DICT = {
 'nose':0,
 'left_eye':1,
 'right_eye':2,
 'left_ear':3,
 'right_ear':4,
 'left_shoulder':5,
 'right_shoulder':6,
 'left_elbow':7,
 'right_elbow':8,
 'left_wrist':9,
 'right_wrist':10,
 'left_hip':11,
 'right_hip':12,
 'left_knee':13,
 'right_knee':14,
 'left_ankle':15,
 'right_ankle':16
}

EDGES = {
    (0, 1): 'm',
    (0, 2): 'c',
    (1, 3): 'm',
    (2, 4): 'c',
    (0, 5): 'm',
    (0, 6): 'c',
    (5, 7): 'm',
    (7, 9): 'm',
    (6, 8): 'c',
    (8, 10): 'c',
    (5, 6): 'y',
    (5, 11): 'm',
    (6, 12): 'c',
    (11, 12): 'y',
    (11, 13): 'm',
    (13, 15): 'm',
    (12, 14): 'c',
    (14, 16): 'c'
}

def draw_keypoints(frame, shaped, confidence_threshold):
    for kp in shaped:
        ky, kx, kp_conf = kp
        if kp_conf > confidence_threshold:
            cv2.circle(frame, (int(kx), int(ky)), 4, (0, 255, 0), -1)

def draw_connections(frame, shaped, edges, confidence_threshold):
    for edge, color in edges.items():
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]

        if (c1 > confidence_threshold) & (c2 > confidence_threshold):
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)


def loop_through_people(frame, keypoints_with_scores, boxes, edges, confidence_threshold):
    count = 0
    frame_h, frame_w, c = frame.shape
    maxArea = 0
    maxPersonBoxHeight = 0
    maxPersonBoxWidth = 0
    maxAreaPersonBoxCenterX = frame_w//2
    maxAreaPersonBoxCenterY = frame_h//2
    
    for person,box in zip(keypoints_with_scores, boxes):
        if (box[4] > 0.1):
            boxupcornerX =int(box[1] * frame_w)
            boxupcornerY =int(box[0] * frame_h)
            boxdowncornerX =int(box[1] * frame_w) + int((box[3] - box[1]) * frame_w)
            boxdowncornerY =int(box[0] * frame_h) + int((box[2] - box[0]) * frame_h)
            cv2.rectangle(frame, (boxupcornerX, boxupcornerY), (boxdowncornerX, boxdowncornerY), (0, 0, 255), 2)
            boxHeight = boxdowncornerY-boxupcornerY
            boxWidth = boxdowncornerX - boxupcornerX
            boxCentreX = int(boxupcornerX + (boxdowncornerX-boxupcornerX)/2)
            boxCentreY = int(boxupcornerY + (boxdowncornerY-boxupcornerY)/2)
            
            y, x, c = frame.shape
            shaped = np.squeeze(np.multiply(person, [y, x, 1]))
            draw_connections(frame, shaped, edges, confidence_threshold)
            draw_keypoints(frame, shaped, confidence_threshold)
            leftShoulder = shaped[5]
            rightShoulder = shaped[6]
            leftShouldery, leftShoulderx, leftShoulder_conf = leftShoulder
            rightShouldery, rightShoulderx, rightShoulder_conf = rightShoulder
            leftAnkle = shaped[15]
            rightAnkle = shaped[16]
            leftAnkley, leftAnklex, leftAnkle_conf = leftAnkle
            rightAnkley, rightAnklex, rightAnkle_conf = rightAnkle
            neckX = leftShoulderx+((rightShoulderx-leftShoulderx)/2)
            neckY = (leftShouldery+rightShouldery)/2
            neckHeight = boxdowncornerY - neckY
            if(leftShoulder_conf > confidence_threshold and rightShoulder_conf > confidence_threshold and leftAnkle_conf>confidence_threshold and rightAnkle_conf>confidence_threshold):
                if(neckY>boxupcornerY+(boxHeight/2)):
                    print("Neck Below Threshold")
                if(boxWidth>boxHeight):
                    print("Lying Down")
                if((neckY>boxupcornerY+(boxHeight/2)) and (boxWidth>boxHeight)):
                    cv2.putText(frame, 'Fall', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
                    print("FALL")
                    cv2.imwrite("C:/xampp/htdocs/tmp/frame%d.jpg" % count, frame)
                    count+=1
                    break

            currentArea = boxHeight*boxWidth
            if currentArea > maxArea:
                maxArea = currentArea
                maxAreaPersonBoxCenterX = boxCentreX
                maxAreaPersonBoxCenterY = boxCentreY
                maxPersonBoxHeight = boxHeight
                maxPersonBoxWidth = boxWidth
                
                maxShaped = np.squeeze(np.multiply(person, [y, x, 1]))
                if (maxShaped[1][2] > confidence_threshold) and (maxShaped[2][2] > confidence_threshold) and (maxShaped[13][2] > confidence_threshold) and (maxShaped[14][2] > confidence_threshold):
                    #print("SUCCESS")
                    #print(person, box)
                    closenessStatus = 1
                
            
    cv2.circle(frame, (maxAreaPersonBoxCenterX, maxAreaPersonBoxCenterY), 5, (255, 0, 0), 2)
    
    
   
    # frameCenter = frame_w//2
    # if maxAreaPersonBoxCenterX < (frameCenter - 100):
    #     locomotionStatus = 4
        
    # elif maxAreaPersonBoxCenterX > (frameCenter + 100):
    #     locomotionStatus = 3
        
    # elif (frameCenter - 150) < maxAreaPersonBoxCenterX < (frameCenter + 150):
    #     locomotionStatus = 1
        
    
    
    return maxAreaPersonBoxCenterX,maxAreaPersonBoxCenterY,count

def detect_fall():
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        img = frame.copy()
        img = cv2.resize(img, (192, 192))
        img = np.expand_dims(img, axis=0)
        input_image = cv2.normalize(src=img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        is_dynamic_shape_model = input_details[0]["shape_signature"][2] == -1
        if is_dynamic_shape_model:
            input_tensor_index = input_details[0]["index"]
            input_shape = input_image.shape
            interpreter.resize_tensor_input(
                input_tensor_index, input_shape, strict=True
            )
        interpreter.allocate_tensors()
        interpreter.set_tensor(input_details[0]['index'], input_image)
        interpreter.invoke()
        keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])
        keypoints = keypoints_with_scores[:, :, :51].reshape((6, 17, 3))
        boxes = keypoints_with_scores[:, :, 51:56]
        panTiltFollowsX, panTiltFollowsY,n = loop_through_people(frame, keypoints, boxes[0], EDGES, 0.1)
        if n:
            break

        cv2.circle(frame, (5, 5), 5, (255, 0, 0), 2)
        cv2.imshow('MoveNet Lightning', frame)
        frame_h, frame_w, c = frame.shape
        print("X "+str(panTiltFollowsX)+" Y "+str(panTiltFollowsY))

        
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


