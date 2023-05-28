import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
model_path ='\models\pose_landmarker_full.task'

counter = 0
stage = None
# BaseOptions = mp.tasks.BaseOptions
# PoseLandmarker = mp.tasks.vision.PoseLandmarker
# PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
# PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult
# VisionRunningMode = mp.tasks.vision.RunningMode

# # Create a pose landmarker instance with the live stream mode:
# def print_result(result: PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
#     print('pose landmarker result: {}'.format(result))

# options = PoseLandmarkerOptions(
#     base_options=BaseOptions(model_asset_path=model_path),
#     running_mode=VisionRunningMode.LIVE_STREAM,
#     result_callback=print_result)

def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
        
    return angle 


cap = cv2.VideoCapture(0)
with mp_pose.Pose(min_detection_confidence = 0.5,min_tracking_confidence = 0.5) as pose:
        
        while True:
            ret, frame = cap.read()
            
            image = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            
            results = pose.process(image)
            
            image.flags.writeable = True
            image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
            
            try:
                landmarks = results.pose_landmarks.landmark
                
                shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                
                # Calculate angle
                angle = calculate_angle(shoulder, elbow, wrist)
                cv2.putText(image, str(angle), 
                            tuple(np.multiply(elbow, [640, 480]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                    )
                
                if angle > 160:
                    stage = "down"
                if angle < 30 and stage =='down':
                    stage="up"
                    counter +=1
                    print(counter)
            except:
                pass
            
            cv2.rectangle(image, (0,0), (225,73), (245,117,16), -1)
            
            # Rep data
            cv2.putText(image, 'REPS', (15,12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, str(counter), 
                        (10,60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
            
            # Stage data
            cv2.putText(image, 'STAGE', (65,12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, stage, 
                        (60,60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
            
            # print(landmark)
            
            mp_drawing.draw_landmarks(image,results.pose_landmarks,mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))
            
            
            
            
        
        
            cv2.imshow('mediapipefeed', image)
            frame = image.tobytes()
            # yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n') 
            code = cv2.waitKey(10)
            if code == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows() 