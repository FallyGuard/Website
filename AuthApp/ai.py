import cv2
import mediapipe as mp
import numpy as np
import math
import requests

from datetime import datetime
import json

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

fallEventAPI = "http://falldetect.somee.com/api/FallEvent"
fallEventDetailsAPI = "http://falldetect.somee.com/api/FallEventDetail"

def model(user, coordinates):
        print("User: ", user['userID'])
        print("Coordinates: ", coordinates)
        
        
        cap = cv2.VideoCapture(0)

        counter = 0 # count the number of falls
        falling_threshold = 100
        standing_threshold = 70
        fall_x = 0 
        fall_y =0
        stage = "No Fall" # 'Fall' or 'no Fall'

        # Adjust the window size 
        cv2.namedWindow("Falling Detection", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Falling Detection", 1320, 680)

        ## Setup mediapipe instance
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
                while cap.isOpened():
                        ret, frame = cap.read()
                        
                        # Recolor image to RGB
                        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        image.flags.writeable = False
                
                        # Make detection
                        results = pose.process(image)
                
                        # Recolor back to BGR
                        image.flags.writeable = True
                        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                        
                        # Extract landmarks
                        try:
                                landmarks = results.pose_landmarks.landmark
                                image_hight, image_width, _ = image.shape
                                
                                # ----------------------   Landmark DOTs   ----------------------           
                                # Normalizing the landmark pixel coordinates the image size
                                # dot - NOSE
                                                                        
                                dot_NOSE_X= int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x * image_width)
                                dot_NOSE_Y= int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y * image_hight)
                                                
                                # dot - LEFT_SHOULDER
                                        
                                dot_LEFT_SHOULDER_X= int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x * image_width)
                                dot_LEFT_SHOULDER_Y= int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y * image_hight)
                                
                                # dot - RIGHT_SHOULDER
                                        
                                dot_RIGHT_SHOULDER_X= int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * image_width)
                                dot_RIGHT_SHOULDER_Y= int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * image_hight)
                                
                                # dot - LEFT_ELBOW
                                        
                                dot_LEFT_ELBOW_X= int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].x * image_width)
                                dot_LEFT_ELBOW_Y= int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].y * image_hight)
                                                
                                # dot - RIGHT_ELBOW
                                        
                                dot_RIGHT_ELBOW_X= int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].x * image_width)
                                dot_RIGHT_ELBOW_Y= int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].y * image_hight)
                                
                                # dot - LEFT_WRIST
                                        
                                dot_LEFT_WRIST_X= int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].x * image_width)
                                dot_LEFT_WRIST_Y= int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y * image_hight)
                                
                                # dot - RIGHT_WRIST
                                        
                                dot_RIGHT_WRIST_X= int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].x * image_width)
                                dot_RIGHT_WRIST_Y= int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].y * image_hight)
                                
                                
                                # dot - LEFT_HIP
                                        
                                dot_LEFT_HIP_X= int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].x * image_width)
                                dot_LEFT_HIP_Y= int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y * image_hight)
                                
                                # dot - RIGHT_HIP
                                        
                                dot_RIGHT_HIP_X= int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].x * image_width)
                                dot_RIGHT_HIP_Y= int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].y * image_hight)
                                
                                # dot - LEFT_KNEE
                                        
                                dot_LEFT_KNEE_X= int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].x * image_width)
                                dot_LEFT_KNEE_Y= int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].y * image_hight)
                                                
                                # dot - RIGHT_KNEE
                                        
                                dot_RIGHT_KNEE_X= int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].x * image_width)
                                dot_RIGHT_KNEE_Y= int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].y * image_hight)
                                

                                # dot - LEFT_ANKLE
                                        
                                dot_LEFT_ANKLE_X= int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].x * image_width)
                                dot_LEFT_ANKLE_Y= int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].y * image_hight)
                                                
                                
                                # dot - RIGHT_ANKLE
                                        
                                dot_RIGHT_ANKLE_X= int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].x * image_width)
                                dot_RIGHT_ANKLE_Y= int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].y * image_hight)
                                
                                # dot - LEFT_HEEL
                                        
                                dot_LEFT_HEEL_X= int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HEEL].x * image_width)
                                dot_LEFT_HEEL_Y= int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HEEL].y * image_hight)
                                
                                
                                # dot - RIGHT_HEEL
                                        
                                dot_RIGHT_HEEL_X= int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HEEL].x * image_width)
                                dot_RIGHT_HEEL_Y= int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HEEL].y * image_hight)
                                
                                                        
                                
                                # dot - LEFT_FOOT_INDEX
                                        
                                dot_LEFT_FOOT_INDEX_X= int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].x * image_width)
                                dot_LEFT_FOOT_INDEX_Y= int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].y * image_hight)
                                
                                
                                # dot - RIGHTFOOT_INDEX
                                        
                                dot_RIGHT_FOOT_INDEX_X= int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].x * image_width)
                                dot_RIGHT_FOOT_INDEX_Y= int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].y * image_hight)
                                
                                # dot - NOSE
                                
                                        
                                dot_NOSE = [ dot_NOSE_X,dot_NOSE_Y]
                                

                                # dot - LEFT_ARM_WRIST_ELBOW
                                
                                dot_LEFT_ARM_A_X = int( (dot_LEFT_WRIST_X+dot_LEFT_ELBOW_X) / 2)
                                dot_LEFT_ARM_A_Y = int( (dot_LEFT_WRIST_Y+dot_LEFT_ELBOW_Y) / 2)
                                
                                LEFT_ARM_WRIST_ELBOW = [dot_LEFT_ARM_A_X,dot_LEFT_ARM_A_Y]
                                
                                
                                # dot - RIGHT_ARM_WRIST_ELBOW
                                
                                dot_RIGHT_ARM_A_X = int( (dot_RIGHT_WRIST_X+dot_RIGHT_ELBOW_X) / 2)
                                dot_RIGHT_ARM_A_Y = int( (dot_RIGHT_WRIST_Y+dot_RIGHT_ELBOW_Y) / 2)
                                
                                RIGHT_ARM_WRIST_ELBOW = [dot_LEFT_ARM_A_X,dot_LEFT_ARM_A_Y]
                                
                                
                                # dot - LEFT_ARM_SHOULDER_ELBOW
                                
                                dot_LEFT_ARM_SHOULDER_ELBOW_X = int( (dot_LEFT_SHOULDER_X+dot_LEFT_ELBOW_X) / 2)
                                dot_LEFT_ARM_SHOULDER_ELBOW_Y = int( (dot_LEFT_SHOULDER_Y+dot_LEFT_ELBOW_Y) / 2)
                                
                                LEFT_ARM_SHOULDER_ELBOW = [    dot_LEFT_ARM_SHOULDER_ELBOW_X   ,     dot_LEFT_ARM_SHOULDER_ELBOW_Y     ]
                                
                                
                                # dot - RIGHT_ARM_SHOULDER_ELBOW
                                
                                dot_RIGHT_ARM_SHOULDER_ELBOW_X = int( (dot_RIGHT_SHOULDER_X+dot_RIGHT_ELBOW_X) / 2)
                                dot_RIGHT_ARM_SHOULDER_ELBOW_Y = int( (dot_RIGHT_SHOULDER_Y+dot_RIGHT_ELBOW_Y) / 2)
                                
                                RIGHT_ARM_SHOULDER_ELBOW = [    dot_RIGHT_ARM_SHOULDER_ELBOW_X   ,     dot_RIGHT_ARM_SHOULDER_ELBOW_Y     ]
                                
                                
                                # dot - BODY_SHOULDER_HIP
                                
                                dot_BODY_SHOULDER_HIP_X = int( (dot_RIGHT_SHOULDER_X+dot_RIGHT_HIP_X+dot_LEFT_SHOULDER_X+dot_LEFT_HIP_X) / 4)
                                dot_BODY_SHOULDER_HIP_Y = int( (dot_RIGHT_SHOULDER_Y+dot_RIGHT_HIP_Y+dot_LEFT_SHOULDER_Y+dot_LEFT_HIP_Y) / 4)
                                
                                BODY_SHOULDER_HIP = [    dot_BODY_SHOULDER_HIP_X   ,     dot_BODY_SHOULDER_HIP_Y     ]
                                
                                
                                # dot - LEFT_LEG_HIP_KNEE
                                
                                dot_LEFT_LEG_HIP_KNEE_X = int( (dot_LEFT_HIP_X+dot_LEFT_KNEE_X) / 2)
                                dot_LEFT_LEG_HIP_KNEE_Y = int( (dot_LEFT_HIP_Y+dot_LEFT_KNEE_Y) / 2)
                                
                                LEFT_LEG_HIP_KNEE = [    dot_LEFT_LEG_HIP_KNEE_X   ,     dot_LEFT_LEG_HIP_KNEE_Y     ]
                                
                                
                                # dot - RIGHT_LEG_HIP_KNEE
                                
                                dot_RIGHT_LEG_HIP_KNEE_X = int( (dot_RIGHT_HIP_X+dot_RIGHT_KNEE_X) / 2)
                                dot_RIGHT_LEG_HIP_KNEE_Y = int( (dot_RIGHT_HIP_Y+dot_RIGHT_KNEE_Y) / 2)
                                
                                RIGHT_LEG_HIP_KNEE = [    dot_RIGHT_LEG_HIP_KNEE_X   ,     dot_RIGHT_LEG_HIP_KNEE_Y     ]
                                
                                
                                # dot - LEFT_LEG_KNEE_ANKLE
                                
                                dot_LEFT_LEG_KNEE_ANKLE_X = int( (dot_LEFT_ANKLE_X+dot_LEFT_KNEE_X) / 2)
                                dot_LEFT_LEG_KNEE_ANKLE_Y = int( (dot_LEFT_ANKLE_Y+dot_LEFT_KNEE_Y) / 2)
                                
                                LEFT_LEG_KNEE_ANKLE = [   dot_LEFT_LEG_KNEE_ANKLE_X   ,     dot_LEFT_LEG_KNEE_ANKLE_Y     ]

                                
                                # dot - RIGHT_LEG_KNEE_ANKLE
                                
                                dot_RIGHT_LEG_KNEE_ANKLE_X = int( (dot_RIGHT_ANKLE_X+dot_RIGHT_KNEE_X) / 2)
                                dot_RIGHT_LEG_KNEE_ANKLE_Y = int( (dot_RIGHT_ANKLE_Y+dot_RIGHT_KNEE_Y) / 2)
                                
                                RIGHT_LEG_KNEE_ANKLE = [   dot_RIGHT_LEG_KNEE_ANKLE_X   ,     dot_RIGHT_LEG_KNEE_ANKLE_Y     ]
                                
                                
                                # dot - LEFT_FOOT_INDEX_HEEL
                                
                                dot_LEFT_FOOT_INDEX_HEEL_X = int( (dot_LEFT_FOOT_INDEX_X+dot_LEFT_HEEL_X) / 2)
                                dot_LEFT_FOOT_INDEX_HEEL_Y = int( (dot_LEFT_FOOT_INDEX_Y+dot_LEFT_HEEL_Y) / 2)
                                
                                LEFT_FOOT_INDEX_HEEL = [    dot_LEFT_FOOT_INDEX_HEEL_X   ,    dot_LEFT_FOOT_INDEX_HEEL_Y    ]
                                
                                                
                                # dot - RIGHT_FOOT_INDEX_HEEL
                                
                                dot_RIGHT_FOOT_INDEX_HEEL_X = int( (dot_RIGHT_FOOT_INDEX_X+dot_RIGHT_HEEL_X) / 2)
                                dot_RIGHT_FOOT_INDEX_HEEL_Y = int( (dot_RIGHT_FOOT_INDEX_Y+dot_RIGHT_HEEL_Y) / 2)
                                
                                RIGHT_FOOT_INDEX_HEEL = [    dot_RIGHT_FOOT_INDEX_HEEL_X   ,    dot_RIGHT_FOOT_INDEX_HEEL_Y    ]
                                
                                
                                # dot _ UPPER_BODY 
                                
                                dot_UPPER_BODY_X = int((dot_NOSE_X+dot_LEFT_ARM_A_X+dot_RIGHT_ARM_A_X+dot_LEFT_ARM_SHOULDER_ELBOW_X+dot_RIGHT_ARM_SHOULDER_ELBOW_X+dot_BODY_SHOULDER_HIP_X)/6)
                                dot_UPPER_BODY_Y = int((dot_NOSE_Y+dot_LEFT_ARM_A_Y+dot_RIGHT_ARM_A_Y+dot_LEFT_ARM_SHOULDER_ELBOW_Y+dot_RIGHT_ARM_SHOULDER_ELBOW_Y+dot_BODY_SHOULDER_HIP_Y)/6)
                                
                                
                                UPPER_BODY = [      dot_UPPER_BODY_X    ,     dot_UPPER_BODY_Y      ]
                                
                                                
                                # dot _ LOWER_BODY
                                
                                dot_LOWER_BODY_X = int( (dot_LEFT_LEG_HIP_KNEE_X+dot_RIGHT_LEG_HIP_KNEE_X+dot_LEFT_LEG_KNEE_ANKLE_X+ dot_RIGHT_LEG_KNEE_ANKLE_X+dot_LEFT_FOOT_INDEX_HEEL_X+dot_RIGHT_FOOT_INDEX_HEEL_X )/6 )
                                dot_LOWER_BODY_Y = int( (dot_LEFT_LEG_HIP_KNEE_Y+dot_RIGHT_LEG_HIP_KNEE_Y+dot_LEFT_LEG_KNEE_ANKLE_Y+ dot_RIGHT_LEG_KNEE_ANKLE_Y+dot_LEFT_FOOT_INDEX_HEEL_Y+dot_RIGHT_FOOT_INDEX_HEEL_Y )/6 )
                                
                                
                                LOWER_BODY = [      dot_LOWER_BODY_X    ,     dot_LOWER_BODY_Y      ]
                                
                                # dot _ BODY
                                
                                dot_BODY_X = int( (dot_UPPER_BODY_X + dot_LOWER_BODY_X)/2 )
                                dot_BODY_Y = int( (dot_UPPER_BODY_Y + dot_LOWER_BODY_Y)/2 )
                                
                                BODY = [      dot_BODY_X    ,     dot_BODY_Y      ]
                                

                                # The mid Point betweent The two Feet (the Point of Action)
                                Point_of_action_LEFT_X = int( 
                                        ((dot_LEFT_FOOT_INDEX_X +  dot_LEFT_HEEL_X)/2) )
                                
                                Point_of_action_LEFT_Y = int( 
                                        ((dot_LEFT_FOOT_INDEX_Y+   dot_LEFT_HEEL_Y)/2) )
                                
                                
                                Point_of_action_RIGHT_X = int( 
                                        ((dot_RIGHT_FOOT_INDEX_X +  dot_RIGHT_HEEL_X)/2) )
                                
                                Point_of_action_RIGHT_Y = int( 
                                        ((dot_RIGHT_FOOT_INDEX_Y+   dot_RIGHT_HEEL_Y)/2) )           
                                
                                
                                Point_of_action_X = int ( (Point_of_action_LEFT_X +  Point_of_action_RIGHT_X)/2 )
                                
                                Point_of_action_Y = int ( (Point_of_action_LEFT_Y +  Point_of_action_RIGHT_Y)/2 )
                                
                                Point_of_action = [Point_of_action_X , Point_of_action_Y]
                                


                                
                        #-------------------------------------------VISUALISING------------------------------------------------------------
                                
                                # visualize the Average point betweent the right and left foot
                                cv2.putText(image, str(Point_of_action), 
                                                (Point_of_action_X,Point_of_action_Y) , 
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (204,252,0), 1, cv2.LINE_AA
                                                        )
                                        
                                cv2.circle(image,  (Point_of_action_X , Point_of_action_Y), 6, (0,0,255), -1)
                                
                                

                                
                                # ----------------------------------Visualize the Landmarks dots---------------------------------------------- 
                                # Visualize dot - dot_NOSE          
                                cv2.putText(image, str(dot_NOSE), 
                                                (dot_NOSE_X,dot_NOSE_Y) , 
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (204,252,0), 1, cv2.LINE_AA
                                                        )         
                                cv2.circle(image,  (dot_NOSE_X,dot_NOSE_Y), 5, (204,252,0), -1)
                                

                                # Visualize dot - LEFT_ARM_WRIST_ELBOW        
                                cv2.putText(image, str(LEFT_ARM_WRIST_ELBOW), 
                                                (dot_LEFT_ARM_A_X,dot_LEFT_ARM_A_Y) , 
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (204,252,0), 1, cv2.LINE_AA
                                                        )
                                        
                                cv2.circle(image,  (dot_LEFT_ARM_A_X,dot_LEFT_ARM_A_Y), 5, (204,252,0), -1)
                                
                                
                                # Visualize dot - RIGHT_ARM_WRIST_ELBOW      
                                cv2.putText(image, str(RIGHT_ARM_WRIST_ELBOW), 
                                                (dot_RIGHT_ARM_A_X,dot_RIGHT_ARM_A_Y) , 
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (204,252,0), 1, cv2.LINE_AA
                                                        )
                                        
                                cv2.circle(image,  (dot_RIGHT_ARM_A_X,dot_RIGHT_ARM_A_Y), 5, (204,252,0), -1)
                        
                                
                        
                                # Visualize dot - LEFT_ARM_SHOULDER_ELBOW          
                                cv2.putText(image, str(LEFT_ARM_SHOULDER_ELBOW), 
                                                (dot_LEFT_ARM_SHOULDER_ELBOW_X,dot_LEFT_ARM_SHOULDER_ELBOW_Y) , 
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (204,252,0), 1, cv2.LINE_AA
                                                        )
                                        
                                cv2.circle(image,  (dot_LEFT_ARM_SHOULDER_ELBOW_X,dot_LEFT_ARM_SHOULDER_ELBOW_Y), 5, (204,252,0), -1) 


                                # Visualize dot - RIGHT_ARM_SHOULDER_ELBOW           
                                cv2.putText(image, str(RIGHT_ARM_SHOULDER_ELBOW), 
                                                (dot_RIGHT_ARM_SHOULDER_ELBOW_X,dot_RIGHT_ARM_SHOULDER_ELBOW_Y) , 
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (204,252,0), 1, cv2.LINE_AA
                                                        )
                                        
                                cv2.circle(image,  (dot_RIGHT_ARM_SHOULDER_ELBOW_X,dot_RIGHT_ARM_SHOULDER_ELBOW_Y), 5, (204,252,0), -1)
                        
                        
                                # Visualize dot - BODY_SHOULDER_HIP           
                                cv2.putText(image, str(BODY_SHOULDER_HIP), 
                                                (dot_BODY_SHOULDER_HIP_X,dot_BODY_SHOULDER_HIP_Y) , 
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (204,252,0), 1, cv2.LINE_AA
                                                        )
                                        
                                cv2.circle(image,  (dot_BODY_SHOULDER_HIP_X,dot_BODY_SHOULDER_HIP_Y), 5, (204,252,0), -1)
                                

                                # Visualize dot - LEFT_LEG_HIP_KNEE             
                                cv2.putText(image, str(LEFT_LEG_HIP_KNEE), 
                                                (dot_LEFT_LEG_HIP_KNEE_X    ,    dot_LEFT_LEG_HIP_KNEE_Y) , 
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (204,252,0), 1, cv2.LINE_AA
                                                        )
                                        
                                cv2.circle(image,  (dot_LEFT_LEG_HIP_KNEE_X    ,    dot_LEFT_LEG_HIP_KNEE_Y), 5, (204,252,0), -1)
                        

                                # Visualize dot - RIGHT_LEG_HIP_KNEE            
                                cv2.putText(image, str(RIGHT_LEG_HIP_KNEE), 
                                                (dot_RIGHT_LEG_HIP_KNEE_X    ,    dot_RIGHT_LEG_HIP_KNEE_Y) , 
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (204,252,0), 1, cv2.LINE_AA
                                                        )
                                        
                                cv2.circle(image,  (dot_RIGHT_LEG_HIP_KNEE_X    ,    dot_RIGHT_LEG_HIP_KNEE_Y), 5, (204,252,0), -1)
                                

                                # Visualize dot - LEFT_LEG_KNEE_ANKLE            
                                cv2.putText(image, str(LEFT_LEG_KNEE_ANKLE), 
                                                (dot_LEFT_LEG_KNEE_ANKLE_X    ,    dot_LEFT_LEG_KNEE_ANKLE_Y) , 
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (204,252,0), 1, cv2.LINE_AA
                                                        )
                                        
                                cv2.circle(image,  (dot_LEFT_LEG_KNEE_ANKLE_X    ,    dot_LEFT_LEG_KNEE_ANKLE_Y), 5, (204,252,0), -1)
                                

                                # Visualize dot - RIGHT_LEG_KNEE_ANKLE            
                                cv2.putText(image, str(RIGHT_LEG_KNEE_ANKLE), 
                                                (dot_RIGHT_LEG_KNEE_ANKLE_X    ,    dot_RIGHT_LEG_KNEE_ANKLE_Y) , 
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (204,252,0), 1, cv2.LINE_AA
                                                        )
                                        
                                cv2.circle(image,  (dot_RIGHT_LEG_KNEE_ANKLE_X    ,    dot_RIGHT_LEG_KNEE_ANKLE_Y), 5, (204,252,0), -1)
                                
                                
                                # Visualize dot -   LEFT_FOOT_INDEX_HEEL             
                                cv2.putText(image, str(LEFT_FOOT_INDEX_HEEL), 
                                                (dot_LEFT_FOOT_INDEX_HEEL_X    ,    dot_LEFT_FOOT_INDEX_HEEL_Y) , 
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (204,252,0), 1, cv2.LINE_AA
                                                        )
                                        
                                cv2.circle(image,  (dot_LEFT_FOOT_INDEX_HEEL_X    ,    dot_LEFT_FOOT_INDEX_HEEL_Y), 5, (204,252,0), -1)
                                
                                
                                # Visualize dot -   RIGHT_FOOT_INDEX_HEEL            
                                cv2.putText(image, str(RIGHT_FOOT_INDEX_HEEL), 
                                                (dot_RIGHT_FOOT_INDEX_HEEL_X    ,    dot_RIGHT_FOOT_INDEX_HEEL_Y) , 
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (204,252,0), 1, cv2.LINE_AA
                                                        )
                                        
                                cv2.circle(image,  (dot_RIGHT_FOOT_INDEX_HEEL_X    ,    dot_RIGHT_FOOT_INDEX_HEEL_Y), 5, (204,252,0), -1)
                                

                                # Visualize dot -   UPPER_BODY            
                                cv2.putText(image, str(UPPER_BODY), 
                                                ( dot_UPPER_BODY_X    ,    dot_UPPER_BODY_Y ) , 
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (277,220,0), 1, cv2.LINE_AA
                                                        )
                                        
                                cv2.circle(image,  (dot_UPPER_BODY_X    ,    dot_UPPER_BODY_Y), 9, (0,0,0), -1)
                                
                                
                                # Visualize dot -   LOWER_BODY            
                                cv2.putText(image, str(LOWER_BODY), 
                                                ( dot_LOWER_BODY_X    ,    dot_LOWER_BODY_Y ) , 
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (277,220,0), 1, cv2.LINE_AA
                                                        )
                                        
                                cv2.circle(image,  (dot_LOWER_BODY_X    ,    dot_LOWER_BODY_Y), 9, (277,220,0), -1)


                                # Visualize dot -   BODY            
                                cv2.putText(image, str(BODY), 
                                                ( dot_BODY_X    ,    dot_BODY_Y ) , 
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA
                                                        )
                                        
                                cv2.circle(image,  (dot_BODY_X    ,    dot_BODY_Y), 12, (0,0,255), -1)
                                
                                
                                
                                #--------------------------------------The Falling Logic---------------------------------------------------
                                
                                # Calculating falling distance
                                #-------- (The Average point between the two feet  -  the point of the center of the body)
                                fall_x = int(Point_of_action_X - dot_BODY_X )
                                
                                # the vertical distance between the upper body and the mid point between the feet
                                fall_y = int(Point_of_action_Y - dot_UPPER_BODY_Y)          

                                # Case Falling and Standing
                                falling = abs(fall_x) > falling_threshold 
                                standing = abs(fall_x) < standing_threshold
                        

                                if falling and abs(fall_y) < 30:
                                        stage="Fall"
                                        # counter+=1 # new one
                                        
                                        # Request to the API
                                        # TODO FallEvent
                                        print("================= Fall Event ID: ========================")
                                        responseFallEvent = requests.post(fallEventAPI, json={
                                                "userID": user['userID'],
                                                "fallTimestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                                                "fallLocation": user['country'],
                                                "isEmergencyNotified": True,
                                        })
                                        
                                        print("================= END Fall Event ID: ========================")
                                        print(responseFallEvent.json()["id"])
                                        
                                        response = requests.post(fallEventDetailsAPI, json= {
                                                "fallEventID": responseFallEvent.json()["id"],
                                                "fallLocation_Lat": coordinates["latitute"],
                                                "fallLocation_Long": coordinates["longitute"]
                                        })
                                        
                                        print("RESPONSE: ", response.json())
                                        
                                        

                                if  standing and stage == 'Fall' and abs(fall_y) > 40 :            
                                        counter+=1 # old one
                                        print("Standed up")
                                        stage = "No Fall"
                        except:
                                pass
                        
                        # visualize the window data

                        cv2.rectangle(image, (0,0), (250,100), (245,117,16), -1)

                        cv2.putText(image, str(fall_x), (65,15), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
                        
                        cv2.putText(image, str(fall_y), (200,15), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
                        
                        cv2.putText(image, 'HD :', (15,15), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
                        
                        cv2.putText(image, 'VD :', (150,15), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
                        
                        cv2.putText(image, str(counter ), 
                                (10,80), 
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 2, cv2.LINE_AA)
                        
                        if stage == "No Fall":
                                cv2.putText(image, stage, 
                                (100,80), 
                                cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 1, cv2.LINE_AA)
                        else:
                                cv2.putText(image, stage, 
                                (100,70), 
                                cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,255), 4, cv2.LINE_AA)
                        
                        # Render detections
                        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                                mp_drawing.DrawingSpec(color=(255,255,255), thickness=2, circle_radius=2), 
                                                mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2) 
                                                )               
                        
                        cv2.imshow('Falling Detection', image)
                        cv2.imshow('Falling Detection', image) # new one
                

                        if cv2.waitKey(10) & 0xFF == ord('q'):
                                break

        cap.release()
        cv2.destroyAllWindows()
        
        
__name__ = "ai"