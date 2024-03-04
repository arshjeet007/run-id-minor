import cv2
import mediapipe as mp

# Global variables for managing multiple pose estimators
pose_estimators = []
pose_estimator_dims = []

# Function to compare distances between object boundaries
def compare_dist(dim1, dim2):
    # Your distance comparison logic here
    pass

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose

def detect_poses(image_path):
    global pose_estimators, pose_estimator_dims

    # Load input image
    image = cv2.imread(image_path)

    # Initialize variables for storing results
    detected_poses = []

    # For each detected object in the image
    for object_boundary in detected_objects:
        if len(pose_estimators) == 0: 
            # Create new pose estimator if none exists
            pose = mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6)
            pose_estimators.append(pose)
            pose_estimator_dims.append(object_boundary)
            selected_pose_idx = len(pose_estimators) - 1
        else:
            # Compare object boundary with existing pose estimators
            threshold_for_new = 100
            prev_high_score = 0
            selected_pose_idx_high = 0
            prev_low_score = 1000000000
            selected_pose_idx_low = 0
            
            for idx, dim in enumerate(pose_estimator_dims):
                score = compare_dist(dim, object_boundary)
                if score > prev_high_score:
                    selected_pose_idx_high = idx
                    prev_high_score = score
                if score < prev_low_score:
                    selected_pose_idx_low = idx
                    prev_low_score = score
            
            if prev_high_score > threshold_for_new:
                # Create new pose estimator if score is high
                pose = mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6)
                pose_estimators.append(pose)
                pose_estimator_dims.append(object_boundary)
                selected_pose_idx = len(pose_estimators) - 1
            else:
                # Select existing pose estimator if score is low
                selected_pose_idx = selected_pose_idx_low
        
        # Process pose estimation for the selected pose estimator
        pose_estimator = pose_estimators[selected_pose_idx]
        results = pose_estimator.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Append detected poses to results list
        detected_poses.append(results)

    return detected_poses

# Example usage
detected_objects = [(x_min1, y_min1, box_width1, box_height1), 
                    (x_min2, y_min2, box_width2, box_height2)]

detected_poses = detect_poses('input_image.jpg')

# Process detected poses
for idx, results in enumerate(detected_poses):
    if not results.pose_landmarks:
        continue
    print(
        f'Nose coordinates: ('
        f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x * image_width}, '
        f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y * image_height})'
    )
    # Draw pose landmarks on the image.
    annotated_image = image.copy()
    mp_drawing.draw_landmarks(
        annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    cv2.imwrite(f'annotated_image_{idx}.jpg', annotated_image)
