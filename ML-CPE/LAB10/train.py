from ultralytics import YOLO
import pandas as pd
import shutil
import glob
import yaml
import os

CLASS_NAMES = ['THAI100', 'THAI1000', 'THAI20', 'THAI50', 'THAI500']

def prepare_dataset():
    """
    ตรวจสอบและจัดเตรียมโครงสร้างโฟลเดอร์ dataset/{train,val,test}/{images,labels}
    รวมถึงสร้างไฟล์ dataset/data.yaml ให้ตรงตามโครงสร้างใบงานอัตโนมัติ
    """
    dataset_dir = "dataset"
    valid_dir = os.path.join(dataset_dir, "valid")
    val_dir = os.path.join(dataset_dir, "val")
    test_dir = os.path.join(dataset_dir, "test")

    # 1. หากโฟลเดอร์ชื่อ valid (จาก Roboflow) ให้เปลี่ยนชื่อเป็น val
    if os.path.exists(valid_dir) and not os.path.exists(val_dir):
        os.rename(valid_dir, val_dir)
        print("[INFO] เปลี่ยนชื่อโฟลเดอร์ 'dataset/valid' เป็น 'dataset/val' เรียบร้อยแล้ว")

    # 2. กรณีผู้ใช้ใส่เฉพาะรูปภาพคู่กับ _classes.csv ให้สร้างโฟลเดอร์ images/labels และไฟล์พิกัด YOLO อัตโนมัติ
    for split in ["train", "val", "test"]:
        split_path = os.path.join(dataset_dir, split)
        csv_path = os.path.join(split_path, "_classes.csv")
        img_dir = os.path.join(split_path, "images")
        lbl_dir = os.path.join(split_path, "labels")

        if os.path.exists(csv_path):
            os.makedirs(img_dir, exist_ok=True)
            os.makedirs(lbl_dir, exist_ok=True)
            df_cls = pd.read_csv(csv_path)
            df_cls.columns = df_cls.columns.str.strip()

            for _, row in df_cls.iterrows():
                fname = str(row["filename"]).strip()
                src_img = os.path.join(split_path, fname)
                dst_img = os.path.join(img_dir, fname)
                if os.path.exists(src_img) and not os.path.exists(dst_img):
                    shutil.move(src_img, dst_img)

                # สร้างไฟล์พิกัด YOLO (.txt) หากยังไม่มี
                txt_name = os.path.splitext(fname)[0] + ".txt"
                txt_path = os.path.join(lbl_dir, txt_name)
                if not os.path.exists(txt_path):
                    lines = []
                    for idx, cname in enumerate(CLASS_NAMES):
                        if cname in df_cls.columns and int(row[cname]) == 1:
                            lines.append(f"{idx} 0.500000 0.500000 0.900000 0.900000\n")
                    with open(txt_path, "w", encoding="utf-8") as f:
                        f.writelines(lines)

    # 3. หากยังไม่มีโฟลเดอร์ test ให้แบ่งข้อมูล 50% จาก val มาเป็น test (จะได้ Train 800 / Val 100 / Test 100)
    val_img_dir = os.path.join(val_dir, "images")
    val_lbl_dir = os.path.join(val_dir, "labels")
    test_img_dir = os.path.join(test_dir, "images")
    test_lbl_dir = os.path.join(test_dir, "labels")

    if os.path.exists(val_img_dir) and not os.path.exists(test_img_dir):
        os.makedirs(test_img_dir, exist_ok=True)
        os.makedirs(test_lbl_dir, exist_ok=True)
        val_images = sorted(glob.glob(os.path.join(val_img_dir, "*.*")))
        half_idx = len(val_images) // 2

        for img_path in val_images[half_idx:]:
            fname = os.path.basename(img_path)
            lbl_name = os.path.splitext(fname)[0] + ".txt"
            lbl_path = os.path.join(val_lbl_dir, lbl_name)

            shutil.move(img_path, os.path.join(test_img_dir, fname))
            if os.path.exists(lbl_path):
                shutil.move(lbl_path, os.path.join(test_lbl_dir, lbl_name))
        print(f"[INFO] แบ่งข้อมูลสร้างโฟลเดอร์ 'dataset/test' จำนวน {len(val_images) - half_idx} ภาพเรียบร้อยแล้ว")

    # 4. สร้างหรืออัปเดตไฟล์ dataset/data.yaml ให้ตรงกับคลาสธนบัตรไทย
    data_yaml_path = os.path.join(dataset_dir, "data.yaml")
    yaml_content = {
        "path": os.path.abspath(dataset_dir),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": 5,
        "names": {i: name for i, name in enumerate(CLASS_NAMES)}
    }
    with open(data_yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(yaml_content, f, sort_keys=False, allow_unicode=True)
    print(f"[INFO] อัปเดตไฟล์ '{data_yaml_path}' สำหรับ 5 คลาส ({', '.join(CLASS_NAMES)}) เรียบร้อยแล้ว\n")
    return data_yaml_path

def main():
    data_yaml_path = prepare_dataset()

    # กำหนดชุดการทดลองเพื่อเปรียบเทียบจำนวน Epochs, Image Size และ Batch Size
    experiments = [
        {"name": "exp1_ep10_img640_b16", "epochs": 10, "imgsz": 640, "batch": 16},
        {"name": "exp2_ep25_img640_b16", "epochs": 25, "imgsz": 640, "batch": 16},
        {"name": "exp3_ep25_img416_b8",  "epochs": 25, "imgsz": 416, "batch": 8},
    ]

    loss_records = []

    for exp in experiments:
        print("=" * 70)
        print(f"เริ่มฝึกสอนโมเดล: {exp['name']}")
        print(f"การตั้งค่า -> Epochs: {exp['epochs']} | Image Size: {exp['imgsz']} | Batch Size: {exp['batch']}")
        print("=" * 70)

        # โหลด Pretrained YOLO Model
        model = YOLO("yolov8n.pt")

        # ฝึกสอนโมเดล
        model.train(
            data=data_yaml_path,
            epochs=exp["epochs"],
            imgsz=exp["imgsz"],
            batch=exp["batch"],
            project="runs/detect",
            name=exp["name"],
            exist_ok=True,
            plots=True
        )

        # ดึงค่า Training และ Validation Loss จากไฟล์ results.csv
        results_csv = os.path.join("runs", "detect", exp["name"], "results.csv")
        if os.path.exists(results_csv):
            df_res = pd.read_csv(results_csv)
            df_res.columns = df_res.columns.str.strip()
            last_row = df_res.iloc[-1]
            loss_records.append({
                "Configuration": exp["name"],
                "Epochs": exp["epochs"],
                "Img_Size": exp["imgsz"],
                "Batch_Size": exp["batch"],
                "Train_Box_Loss": round(float(last_row.get("train/box_loss", 0)), 4),
                "Train_Cls_Loss": round(float(last_row.get("train/cls_loss", 0)), 4),
                "Val_Box_Loss": round(float(last_row.get("val/box_loss", 0)), 4),
                "Val_Cls_Loss": round(float(last_row.get("val/cls_loss", 0)), 4),
            })

    if loss_records:
        loss_df = pd.DataFrame(loss_records)
        print("\n" + "=" * 80)
        print("ตารางสรุปผล Training Loss และ Validation Loss ของแต่ละ Configuration (Thai BankNotes):")
        print("=" * 80)
        print(loss_df.to_string(index=False))
        loss_df.to_csv("training_loss_comparison.csv", index=False)
        print("\n[SUCCESS] บันทึกตารางเปรียบเทียบ Loss ลงในไฟล์ 'training_loss_comparison.csv' เรียบร้อยแล้ว")

if __name__ == "__main__":
    main()