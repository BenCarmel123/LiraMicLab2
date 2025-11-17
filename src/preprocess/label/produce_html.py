def csv_to_html(csv_path, images_dir, output_html):
    from collections import defaultdict
    import csv, os

    subject_map = defaultdict(list)

    # Read CSV safely
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        next(reader)  # skip header

        for row in reader:
            # skip blank or malformed lines
            if len(row) < 2:
                continue

            filename, angle_label = row

            # skip summary lines at bottom
            if filename.startswith("subjects_"):
                continue

            if filename.strip() == "":
                continue

            subject_id = int(filename.split("_")[0])
            subject_map[subject_id].append((filename, angle_label))

    # --- BUILD HTML ---
    html = []
    html.append("""
    <html>
    <head>
        <style>
            .subject-block {
                border: 1px solid #ccc;
                padding: 10px;
                margin-bottom: 20px;
                width: fit-content;
            }
            .img-row {
                display: flex;
                gap: 10px;
            }
            .img-container {
                text-align: center;
                font-family: Arial, sans-serif;
            }
            img {
                height: 180px;
                border: 1px solid #888;
            }
            h2 {
                font-family: Arial, sans-serif;
            }
        </style>
    </head>
    <body>
    <h1>KDEF Angle Verification</h1>
    """)

    angle_text = {"0": "Front", "1": "Right", "2": "Left", "None": "None"}

    for subject_id in sorted(subject_map.keys()):
        imgs = subject_map[subject_id]

        html.append(f"<div class='subject-block'>")
        html.append(f"<h2>Subject {subject_id}</h2>")
        html.append("<div class='img-row'>")

        for filename, angle in sorted(imgs):
            label = angle_text.get(angle, angle)
            img_path = os.path.join(images_dir, filename)

            html.append(f"""
                <div class="img-container">
                    <img src="{img_path}" alt="{filename}">
                    <div><b>{label}</b></div>
                </div>
            """)

        html.append("</div></div>")

    html.append("</body></html>")

    with open(output_html, "w") as f:
        f.write("\n".join(html))

    print(f"HTML visualization saved to {output_html}")
