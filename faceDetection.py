import cv2
import numpy as np

# Load Haar Cascade classifiers for face detection
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# Hand gesture detection parameters
HAND_AREA_THRESHOLD = 500

def detect_faces_with_expressions_webcam():
    """
    Detects faces, expressions, and hand gestures from webcam using OpenCV.
    Works better for VR scenarios (hand detection is reliable, face detection optional).
    Press 'q' to exit.
    """
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Cannot open webcam")
        return
    
    print("Face & Hand Gesture Detection Started. Press 'q' to exit.")
    print("Note: Face detection may not work well with VR headsets. Hand detection works better.")
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Failed to read frame")
            break
        
        # Flip the frame horizontally for selfie-view
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces (may fail with VR headset)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        # Draw face detections and analyze expressions
        for i, (x, y, w, h) in enumerate(faces):
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Extract face region
            face_roi = frame[y:y+h, x:x+w]
            
            # Analyze expression
            expression = analyze_expression_from_roi(face_roi)
            
            # Draw labels
            cv2.putText(frame, f'Face {i+1}',
                      (x, y - 30),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            cv2.putText(frame, f'Expression: {expression}',
                      (x, y - 10),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Detect hand gestures (works well with VR)
        hand_detections = detect_hand_gestures(frame)
        
        # Draw hand detections
        for i, hand in enumerate(hand_detections):
            x, y, w, h = hand['bbox']
            gesture = hand['gesture']
            confidence = hand['confidence']
            
            # Draw bounding box for hand
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            # Draw gesture label
            cv2.putText(frame, f'Hand {i+1}: {gesture}',
                      (x, y - 10),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            cv2.putText(frame, f'Confidence: {confidence:.2f}',
                      (x, y + h + 20),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 2)
            
            # Draw contour
            cv2.drawContours(frame, [hand['contour']], 0, (255, 100, 0), 2)
        
        # Detect VR controllers (by color - typically black/white/bright colors)
        vr_controllers = detect_vr_controllers(frame)
        
        # Draw VR controller detections
        for i, controller in enumerate(vr_controllers):
            x, y, w, h = controller['bbox']
            
            # Draw bounding box for controller
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 165, 255), 2)
            
            # Draw label
            cv2.putText(frame, f'Controller {i+1}',
                      (x, y - 10),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
        
        # Display statistics
        status = f'Faces: {len(faces)} | Hands: {len(hand_detections)} | Controllers: {len(vr_controllers)}'
        cv2.putText(frame, status,
                  (10, 30),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display the frame
        cv2.imshow('Face & Hand Gesture Detection (VR Compatible)', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


def analyze_expression_from_roi(face_roi):
    """
    Analyzes facial expressions from a face region using histogram analysis.
    Returns a string describing the detected expression.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
    
    # Calculate histogram
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    
    # Calculate brightness metrics
    brightness = np.mean(gray)
    contrast = np.std(gray)
    
    # Calculate edge density (potential movement/expression)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges) / edges.size
    
    # Simple expression detection based on histogram and contrast
    if brightness > 150 and contrast > 50:
        return "Happy/Bright"
    elif contrast < 30:
        return "Neutral/Calm"
    elif edge_density > 0.1:
        return "Surprised/Expressive"
    else:
        return "Neutral"


def detect_hand_gestures(frame):
    """
    Detects hands and recognizes gestures from the frame.
    Returns list of tuples (hand_bbox, gesture_name, confidence)
    """
    # Convert to HSV for better skin detection
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
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    hand_detections = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # Filter by area
        if area < HAND_AREA_THRESHOLD:
            continue
        
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        
        # Filter by aspect ratio (hands are roughly square-ish)
        aspect_ratio = float(w) / h if h > 0 else 0
        if aspect_ratio > 3 or aspect_ratio < 0.3:
            continue
        
        # Get convex hull
        hull = cv2.convexHull(contour)
        
        # Analyze hand gesture
        gesture = analyze_hand_gesture(contour, hull, area)
        
        # Calculate confidence (based on area and contour complexity)
        confidence = min(area / 5000, 1.0)
        
        hand_detections.append({
            'bbox': (x, y, w, h),
            'gesture': gesture,
            'confidence': confidence,
            'contour': contour,
            'hull': hull
        })
    
    return hand_detections


def analyze_hand_gesture(contour, hull, area):
    """
    Analyzes hand contour to recognize gestures.
    Returns gesture name.
    """
    # Approximate the contour
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    
    # Count convexity defects
    hull_indices = cv2.convexHull(contour, returnPoints=False)
    
    if len(hull_indices) > 3:
        defects = cv2.convexityDefects(contour, hull_indices)
        
        if defects is not None:
            # Count valid defects (potential finger valleys)
            valid_defects = 0
            for defect in defects:
                start, end, farthest, depth = defect
                if depth > 1000:  # Threshold for valid defects
                    valid_defects += 1
            
            # Estimate number of raised fingers based on defects
            num_fingers = valid_defects + 1
            
            # Classify gesture by number of fingers
            if num_fingers <= 1:
                return "Fist"
            elif num_fingers <= 2:
                return "Peace/Two Fingers"
            elif num_fingers <= 3:
                return "Three Fingers"
            elif num_fingers <= 4:
                return "Four Fingers"
            else:
                return "Open Hand"
    
    # Fallback based on hull size
    hull_area = cv2.contourArea(hull) if len(hull) > 0 else 0
    solidity = float(area) / hull_area if hull_area > 0 else 0
    
    if solidity > 0.8:
        return "Fist"
    else:
        return "Open Hand"


def detect_vr_controllers(frame):
    """
    Detects VR controllers by looking for high-contrast regions.
    Controllers typically have bright LEDs or distinctive shapes.
    Returns list of controller detections.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply threshold to find high-contrast areas (potential controllers)
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    
    # Also detect dark areas (common controller color)
    _, binary_dark = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    
    # Combine both
    combined = cv2.bitwise_or(binary, binary_dark)
    
    # Apply morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    controller_detections = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # Filter by area (controllers are typically medium-sized)
        if area < 500 or area > 50000:
            continue
        
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        
        # Filter by aspect ratio (controllers are elongated)
        aspect_ratio = float(w) / h if h > 0 else 0
        if aspect_ratio > 0.3 and aspect_ratio < 3:  # Allow some flexibility
            confidence = min(area / 15000, 1.0)
            
            controller_detections.append({
                'bbox': (x, y, w, h),
                'confidence': confidence,
                'contour': contour
            })
    
    return controller_detections


def detect_faces_from_image(image_path):
    """
    Detects faces, expressions, hand gestures, and VR controllers from a static image.
    Works for VR scenarios.
    
    Args:
        image_path (str): Path to the image file
    """
    # Read the image
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Error: Cannot read image from {image_path}")
        return
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    
    # Draw face detections and analyze expressions
    for i, (x, y, w, h) in enumerate(faces):
        # Draw bounding box
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Extract face region
        face_roi = image[y:y+h, x:x+w]
        
        # Analyze expression
        expression = analyze_expression_from_roi(face_roi)
        
        # Draw labels
        cv2.putText(image, f'Face {i+1} - {expression}',
                  (x, y - 10),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Detect hand gestures
    hand_detections = detect_hand_gestures(image)
    
    # Draw hand detections
    for i, hand in enumerate(hand_detections):
        x, y, w, h = hand['bbox']
        gesture = hand['gesture']
        
        # Draw bounding box for hand
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        # Draw gesture label
        cv2.putText(image, f'Hand {i+1}: {gesture}',
                  (x, y - 10),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        # Draw contour
        cv2.drawContours(image, [hand['contour']], 0, (255, 100, 0), 2)
    
    # Detect VR controllers
    vr_controllers = detect_vr_controllers(image)
    
    # Draw VR controller detections
    for i, controller in enumerate(vr_controllers):
        x, y, w, h = controller['bbox']
        
        # Draw bounding box for controller
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 165, 255), 2)
        
        # Draw label
        cv2.putText(image, f'Controller {i+1}',
                  (x, y - 10),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
    
    print(f"Detected {len(faces)} face(s), {len(hand_detections)} hand(s), {len(vr_controllers)} controller(s)")
    
    # Display the image
    cv2.imshow('Face & Hand Gesture Detection Results (VR Compatible)', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Choose mode
    print("Face Detection, Expression & Hand Gesture Analysis")
    print("=" * 50)
    print("VR COMPATIBLE MODE: Detects hands, gestures, and VR controllers")
    print("=" * 50)
    print("1. Webcam Detection (Face + Expression + Hands + VR Controllers)")
    print("2. Image Detection (Face + Expression + Hands + VR Controllers)")
    
    choice = input("Enter your choice (1 or 2): ").strip()
    
    if choice == "1":
        detect_faces_with_expressions_webcam()
    elif choice == "2":
        image_path = input("Enter the path to the image file: ").strip()
        detect_faces_from_image(image_path)
    else:
        print("Invalid choice. Please enter 1 or 2.")
