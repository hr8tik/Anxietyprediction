"""
AI-Based Movement Analysis for Pediatric Dentistry Study

Tracks:
1. Eye Blink Frequency
2. Head Movement
3. Facial Movement
4. Hand Movement
5. Body Movement

Output:
CSV file containing movement metrics.

Author: Research Prototype
Modified: Works with OpenCV (compatible with all MediaPipe versions)
"""

import cv2
import pandas as pd
import numpy as np
from math import sqrt
import os

# -----------------------------------------
# LOAD CASCADE CLASSIFIERS
# -----------------------------------------

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_eye.xml'
)

# -----------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------

def euclidean(p1, p2):
    """Calculate Euclidean distance between two points"""
    return sqrt(
        (p1[0] - p2[0])**2 +
        (p1[1] - p2[1])**2
    )

def get_center(x, y, w, h):
    """Get center point of a bounding box"""
    return (x + w//2, y + h//2)

# -----------------------------------------
# MAIN ANALYSIS FUNCTION
# -----------------------------------------

def analyze_video(video_path):
    """
    Analyze a video file and extract movement metrics.
    
    Args:
        video_path (str): Path to the video file
        
    Returns:
        str: Path to the output CSV file with movement metrics
        
    Raises:
        FileNotFoundError: If video file doesn't exist
        IOError: If video cannot be opened
    """
    
    # Check if video file exists
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file '{video_path}' not found.")
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise IOError(f"Cannot open video file '{video_path}'")
    
    # -----------------------------------------
    # VARIABLES FOR METRICS
    # -----------------------------------------
    
    frame_count = 0
    eye_blinks = 0
    previous_eye_distance = None
    head_movements = 0
    previous_face_center = None
    hand_movements = 0
    previous_hand_center = None
    body_movements = 0
    face_area_history = []
    
    # -----------------------------------------
    # PROCESS VIDEO
    # -----------------------------------------
    
    print("Processing video...")
    
    while cap.isOpened():
        success, frame = cap.read()
        
        if not success:
            break
        
        frame_count += 1
        
        # Show progress every 30 frames
        if frame_count % 30 == 0:
            print(f"Frame {frame_count}...", end='\r')
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # ---------------------------------
        # FACE ANALYSIS
        # ---------------------------------
        
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        if len(faces) > 0:
            # Get the largest face (assuming main subject)
            largest_face = max(faces, key=lambda f: f[2] * f[3])
            x, y, fw, fh = largest_face
            
            # Calculate face center
            face_center = get_center(x, y, fw, fh)
            
            # HEAD MOVEMENT
            if previous_face_center is not None:
                movement = euclidean(face_center, previous_face_center)
                if movement > 10:
                    head_movements += 1
            
            previous_face_center = face_center
            
            # BLINK DETECTION - detect eyes within face
            roi_gray = gray[y:y+fh, x:x+fw]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            
            if len(eyes) >= 2:
                # Get top 2 eyes (likely left and right eyes)
                eyes = sorted(eyes, key=lambda e: e[1])[:2]
                
                # Calculate average eye vertical distance
                eye_distances = []
                for (ex, ey, ew, eh) in eyes:
                    eye_distance = eh  # Vertical height of eye
                    eye_distances.append(eye_distance)
                
                avg_eye_distance = np.mean(eye_distances) if eye_distances else 0
                
                # BLINK DETECTION
                if previous_eye_distance is not None:
                    # Blink detected when eye distance drops significantly
                    if avg_eye_distance < previous_eye_distance * 0.5 and previous_eye_distance > 5:
                        eye_blinks += 1
                
                previous_eye_distance = avg_eye_distance
        
        # ---------------------------------
        # HAND ANALYSIS (Skin Detection)
        # ---------------------------------
        
        # Convert to HSV for skin detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Define skin color range in HSV
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        
        # Create mask for skin detection
        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
        # Also detect other skin tones
        lower_skin2 = np.array([170, 20, 70], dtype=np.uint8)
        upper_skin2 = np.array([180, 255, 255], dtype=np.uint8)
        mask2 = cv2.inRange(hsv, lower_skin2, upper_skin2)
        
        # Combine masks
        mask = cv2.bitwise_or(mask, mask2)
        
        # Apply morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter hands by contour area
        hands = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if 500 < area < 50000:  # Filter by reasonable hand size
                x, y, w, h = cv2.boundingRect(contour)
                hands.append((x, y, w, h))
        
        # Process detected hands
        if len(hands) > 0:
            # Get the largest hand
            largest_hand = max(hands, key=lambda h: h[2] * h[3])
            hx, hy, hw, hh = largest_hand
            
            hand_center = get_center(hx, hy, hw, hh)
            
            # HAND MOVEMENT
            if previous_hand_center is not None:
                movement = euclidean(hand_center, previous_hand_center)
                if movement > 15:
                    hand_movements += 1
            
            previous_hand_center = hand_center
        
        # ---------------------------------
        # BODY/FACIAL MOVEMENT (using face area variation)
        # ---------------------------------
        
        if len(faces) > 0:
            largest_face = max(faces, key=lambda f: f[2] * f[3])
            x, y, fw, fh = largest_face
            face_area = fw * fh
            
            face_area_history.append(face_area)
            
            # Detect movement based on face area changes
            if len(face_area_history) > 2:
                recent_areas = face_area_history[-3:]
                area_variance = np.var(recent_areas)
                
                # Movement detected if significant area variance
                if area_variance > 5000:
                    body_movements += 1
    
    # -----------------------------------------
    # RELEASE VIDEO
    # -----------------------------------------
    
    cap.release()
    cv2.destroyAllWindows()
    
    # -----------------------------------------
    # CALCULATE METRICS
    # -----------------------------------------
    
    fps = 30  # Assuming standard 30 fps
    
    # Get actual FPS from video
    cap_check = cv2.VideoCapture(video_path)
    actual_fps = cap_check.get(cv2.CAP_PROP_FPS)
    cap_check.release()
    
    if actual_fps > 0:
        fps = actual_fps
    
    duration_seconds = frame_count / fps
    duration_minutes = duration_seconds / 60
    
    # Calculate blink rate (per minute)
    if duration_minutes > 0:
        blink_rate = eye_blinks / duration_minutes
    else:
        blink_rate = 0
    
    # -----------------------------------------
    # CREATE OUTPUT DATASET
    # -----------------------------------------
    
    results = pd.DataFrame({
        "Frame_Count": [frame_count],
        "Duration_Seconds": [duration_seconds],
        "Duration_Minutes": [duration_minutes],
        "Eye_Blinks": [eye_blinks],
        "Blink_Rate_Per_Minute": [blink_rate],
        "Head_Movements": [head_movements],
        "Hand_Movements": [hand_movements],
        "Body_Movements": [body_movements]
    })
    
    # -----------------------------------------
    # CREATE OUTPUT DIRECTORY
    # -----------------------------------------
    
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # -----------------------------------------
    # SAVE CSV
    # -----------------------------------------
    
    output_path = os.path.join(output_dir, "movement_data.csv")
    results.to_csv(output_path, index=False)
    
    print("\n" + "="*50)
    print("Analysis Complete!")
    print("="*50)
    print(results.to_string(index=False))
    
    return output_path
