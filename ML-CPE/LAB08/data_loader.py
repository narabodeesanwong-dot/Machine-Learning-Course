import os
import json
import numpy as np
import pandas as pd

def load_dataset(csv_path="../WineDataset/winequality-red.csv.csv", output_dir="outputs", export_class_folders=True):
    os.makedirs(output_dir, exist_ok=True)

    # โหลดข้อมูลและข้ามแถวที่เสียหายหรือมีค่าว่าง (Skip corrupted rows)
    df = pd.read_csv(csv_path)
    df = df.dropna().reset_index(drop=True)

    feature_names = df.drop("quality", axis=1).columns.tolist()
    X = df.drop("quality", axis=1).values.astype(np.float32)
    y_raw = df["quality"].values.astype(int)

    # เรียงลำดับ Class จากน้อยไปมาก: Quality_3, Quality_4, ..., Quality_8
    unique_classes = sorted(np.unique(y_raw))
    class_names = [f"Quality_{c}" for c in unique_classes]
    label_map = {c: idx for idx, c in enumerate(unique_classes)}
    y_encoded = np.array([label_map[val] for val in y_raw], dtype=np.int32)

    # สร้างโฟลเดอร์แยกตาม Class เหมือนตัวอย่างโครงสร้าง Dataset
    if export_class_folders:
        base_folder = "WineDataset"
        for c in unique_classes:
            cls_folder = os.path.join(base_folder, f"Quality_{c}")
            os.makedirs(cls_folder, exist_ok=True)
            cls_df = df[df["quality"] == c]
            cls_df.to_csv(os.path.join(cls_folder, f"samples_quality_{c}.csv"), index=False)

    # บันทึก labels.npy และ classes.json
    np.save(os.path.join(output_dir, "labels.npy"), y_encoded)
    classes_dict = {str(i): name for i, name in enumerate(class_names)}
    with open(os.path.join(output_dir, "classes.json"), "w", encoding="utf-8") as f:
        json.dump(classes_dict, f, indent=4)

    print(f"[Data Loader] Loaded {len(df)} valid rows.")
    print(f"[Data Loader] Classes sorted: {classes_dict}")
    return X, y_encoded, class_names, feature_names