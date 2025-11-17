import cv2
import mediapipe as mp
import os
from collections import defaultdict
import numpy as np
mp_face_mesh = mp.solutions.face_mesh

# --- Estimate angle from landmarks ---

def estimate_angle_from_landmarks(landmarks):
    """
    Estimate face orientation from MediaPipe 468 face landmarks.
    
    Args:
        landmarks: MediaPipe face landmarks (468 points with x, y, z coordinates)
        
    Returns:
        int: 0 = frontal, 1 = right profile, 2 = left profile
    """
    
    # Key landmark indices
    NOSE_TIP = 1
    LEFT_EYE_OUTER = 33
    RIGHT_EYE_OUTER = 263
    LEFT_CHEEK = 234
    RIGHT_CHEEK = 454
    
    # Extract x coordinates
    nose_x = landmarks[NOSE_TIP].x
    left_eye_x = landmarks[LEFT_EYE_OUTER].x
    right_eye_x = landmarks[RIGHT_EYE_OUTER].x
    
    # Calculate eye center and nose offset
    eye_center_x = (left_eye_x + right_eye_x) / 2
    eye_width = abs(right_eye_x - left_eye_x)
    nose_offset = nose_x - eye_center_x
    
    # Calculate yaw angle in degrees
    if eye_width > 0.001:
        yaw_angle = np.degrees(np.arctan2(nose_offset, eye_width / 2))
    else:
        yaw_angle = 0
    
    # Determine orientation and return 0, 1, or 2
    if abs(yaw_angle) < 15:
        return 0  # Frontal
    elif yaw_angle > 0:
        return 1  # Right profile
    else:
        return 2  # Left profile
    
# --- Process a single image ---
def get_face_angle(image_path, face_mesh):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if not results.multi_face_landmarks:
        return None

    landmarks = results.multi_face_landmarks[0].landmark
    return estimate_angle_from_landmarks(landmarks)

# --- Process directory using ONE FaceMesh instance ---
def process_image_directory(input_dir):
    angle_results = {}

    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    ) as face_mesh:

        for fname in os.listdir(input_dir):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            path = os.path.join(input_dir, fname)
            angle = get_face_angle(path, face_mesh)
            angle_results[fname] = angle

    return angle_results

# --- Save output to CSV with summary ---
def label_angles_in_directory(input_dir, output_file):
    """Process images, write angle labels, and append summary of subjects missing 0-1-2."""
    
    angle_results = process_image_directory(input_dir)

    # Sort by subject ID
    sorted_items = sorted(
        angle_results.items(),
        key=lambda item: int(item[0].split("_")[0])
    )

    # Group by subject ID
    subject_to_angles = defaultdict(list)
    for fname, angle in sorted_items:
        subject_id = int(fname.split("_")[0])
        subject_to_angles[subject_id].append(angle)

    # Count how many subjects do NOT have a full triplet
    total_subjects = len(subject_to_angles)
    bad_subjects = sum(1 for angles in subject_to_angles.values() if set(angles) != {0, 1, 2})

    # percentage
    bad_percent = (bad_subjects / total_subjects) * 100 if total_subjects > 0 else 0

    # Write CSV
    with open(output_file, 'w') as f:
        f.write("filename,angle_label\n")

        for fname, angle in sorted_items:
            angle_label = "None" if angle is None else str(angle)
            f.write(f"{fname},{angle_label}\n")

        # Append summary
        f.write("\n")
        f.write(f"subjects_missing_full_0_1_2_triplet,{bad_subjects}\n")
        f.write(f"subjects_missing_percent,{bad_percent:.2f}%\n")

    print(f"Angle labels written to {output_file}")
    print(f"Subjects missing 0-1-2 triplet: {bad_subjects} ({bad_percent:.2f}%)")
