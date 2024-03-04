import cv2
import mediapipe as mp

# Initialize Mediapipe Pose Detection solution
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Load input image
image = cv2.imread('pic1.jpeg')

# Convert the image to RGB (Mediapipe requires RGB input)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Run pose detection on the image
results = pose.process(image_rgb)

# Check if any pose was detected
if results.pose_landmarks:
    # Access the landmarks (keypoints) of the detected pose
    pose_landmarks = results.pose_landmarks.landmark
    
    # Get the indices of the shoulder and waist keypoints
    left_shoulder = pose_landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = pose_landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    left_hip = pose_landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
    right_hip = pose_landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]

    # Convert normalized coordinates to pixel coordinates
    image_height, image_width, _ = image.shape
    left_shoulder_x = int(left_shoulder.x * image_width)
    left_shoulder_y = int(left_shoulder.y * image_height)
    right_shoulder_x = int(right_shoulder.x * image_width)
    right_shoulder_y = int(right_shoulder.y * image_height)
    left_hip_x = int(left_hip.x * image_width)
    left_hip_y = int(left_hip.y * image_height)
    right_hip_x = int(right_hip.x * image_width)
    right_hip_y = int(right_hip.y * image_height)

    # Calculate the minimum and maximum coordinates for the bounding box
    min_x = min(left_shoulder_x, right_shoulder_x, left_hip_x, right_hip_x)
    min_y = min(left_shoulder_y, right_shoulder_y, left_hip_y, right_hip_y)
    max_x = max(left_shoulder_x, right_shoulder_x, left_hip_x, right_hip_x)
    max_y = max(left_shoulder_y, right_shoulder_y, left_hip_y, right_hip_y)

    # Draw a red bounding box around the shoulder and waist points
    cv2.rectangle(image, (min_x, min_y), (max_x, max_y), (0, 0, 255), 2)

# Save the modified image with the bounding box
cv2.imwrite('result_torso.jpg', image)

