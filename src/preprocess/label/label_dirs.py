import os
from detect_angle import label_angles_in_directory
from produce_html import csv_to_html

def process_emotion_directories(base_dir):
    """
    For every subdirectory inside base_dir:
    - Run the angle detection
    - Save CSV inside that subdirectory
    - Produce matching HTML visualization inside that subdirectory
    """

    # List all subfolders (each emotion)
    emotions = [
        name for name in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, name))
    ]

    print("Found emotion folders:", emotions)

    for emotion in emotions:
        emotion_path = os.path.join(base_dir, emotion)
        csv_path = os.path.join(emotion_path, f"{emotion}_angles.csv")
        html_path = os.path.join(emotion_path, f"{emotion}_angles.html")

        print(f"\nProcessing emotion: {emotion}")
        print(f" - Input images: {emotion_path}")
        print(f" - CSV output:   {csv_path}")
        print(f" - HTML output:  {html_path}")

        # 1. Generate CSV with angle predictions
        label_angles_in_directory(input_dir=emotion_path, output_file=csv_path)

        # 2. Generate HTML visualization
        csv_to_html(csv_path=csv_path, images_dir=emotion_path, output_html=html_path)

    print("\nAll emotion folders processed successfully!")


if __name__ == "__main__":
    BASE_DIR = "/Users/bencarmel/Documents/TAU/LiraMic/src/dataset/processed_kdef"
    process_emotion_directories(BASE_DIR)
