import cv2
from mtcnn import MTCNN

# Initialize MTCNN detector
detector = MTCNN()

# Load image
image = cv2.cvtColor(cv2.imread("foot.jpg"), cv2.COLOR_BGR2RGB)

# Detect faces in the image
results = detector.detect_faces(image)

# Iterate over each detected face
for result in results:
    bounding_box = result['box']  # Bounding box coordinates
    keypoints = result['keypoints']  # Facial keypoints

    # Draw bounding box around the face
    cv2.rectangle(image,
                  (bounding_box[0], bounding_box[1]),
                  (bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]),
                  (0, 155, 255),
                  2)

    # Draw facial keypoints
    cv2.circle(image, (keypoints['left_eye']), 2, (0, 155, 255), 2)
    cv2.circle(image, (keypoints['right_eye']), 2, (0, 155, 255), 2)
    cv2.circle(image, (keypoints['nose']), 2, (0, 155, 255), 2)
    cv2.circle(image, (keypoints['mouth_left']), 2, (0, 155, 255), 2)
    cv2.circle(image, (keypoints['mouth_right']), 2, (0, 155, 255), 2)

# Save the modified image
cv2.imwrite("result_face.jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

# Print detection results
print(results)