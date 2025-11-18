
import os
import shutil
import csv

def build_kdef_by_angle(processed_kdef_dir):
    """
    Creates a folder `kdef_by_angle` containing:
        front/left/right/
            emotion1/
            emotion2/
            ...
    Populates them using the CSV angle labels in each emotion directory.
    """

    # ---- Step 1: Create base output directory ----
    base_output = os.path.join(os.path.dirname(processed_kdef_dir), "kdef_by_angle")
    os.makedirs(base_output, exist_ok=True)

    angle_map = {
        "0": "front",
        "1": "right",
        "2": "left"
    }

    emotions = [
        name for name in os.listdir(processed_kdef_dir)
        if os.path.isdir(os.path.join(processed_kdef_dir, name))
    ]

    # ---- Step 2: Create 3×7 folder structure ----
    for angle_name in ["front", "left", "right"]:
        for emotion in emotions:
            out_dir = os.path.join(base_output, angle_name, emotion)
            os.makedirs(out_dir, exist_ok=True)

    print("Folder structure created under:", base_output)

    # ---- Step 3: For each emotion folder, read its CSV ----
    for emotion in emotions:
        emotion_path = os.path.join(processed_kdef_dir, emotion)
        csv_path = os.path.join(emotion_path, f"{emotion}_angles.csv")

        if not os.path.exists(csv_path):
            print(f"WARNING: CSV not found for {emotion}, skipping.")
            continue

        print(f"Processing {emotion} using {csv_path}")

        # Read CSV
        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            header = next(reader)  # "filename", "angle_label"

            for row in reader:
                if len(row) < 2:
                    continue

                filename, angle_label = row

                if filename.startswith("subjects_"):  # skip summary rows
                    continue

                angle_label = angle_label.strip()
                if angle_label not in angle_map:
                    continue  # Skip None or invalid

                angle_folder = angle_map[angle_label]

                src_image = os.path.join(emotion_path, filename)
                dst_image = os.path.join(base_output, angle_folder, emotion, filename)

                if os.path.exists(src_image):
                    shutil.copy(src_image, dst_image)

        print(f"Finished emotion: {emotion}")

    print("\n✔ All images copied successfully into:", base_output)


if __name__ == "__main__":
    build_kdef_by_angle("/Users/bencarmel/Documents/TAU/LiraMic/src/dataset/processed_kdef")