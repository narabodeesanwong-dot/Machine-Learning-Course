from ultralytics import YOLO
import pandas as pd
import os

def evaluate_models():
    data_yaml_path = os.path.join("dataset", "data.yaml")

    experiments = [
        {"name": "exp1_ep10_img640_b16", "epochs": 10, "imgsz": 640, "batch": 16},
        {"name": "exp2_ep25_img640_b16", "epochs": 25, "imgsz": 640, "batch": 16},
        {"name": "exp3_ep25_img416_b8",  "epochs": 25, "imgsz": 416, "batch": 8},
    ]

    eval_records = []

    for exp in experiments:
        weight_path = os.path.join("runs", "detect", "runs", "detect", exp["name"], "weights", "best.pt")
        if not os.path.exists(weight_path):
            print(f"[WARNING] ไม่พบไฟล์โมเดล {weight_path} (กรุณารัน train.py ก่อน)")
            continue

        print(f"\nกำลังประเมินประสิทธิภาพโมเดล: {exp['name']} ...")
        model = YOLO(weight_path)

        # ประเมินผลด้วยชุดข้อมูล test (หากไม่มีโฟลเดอร์ test จะสลับใช้ val อัตโนมัติ)
        split_target = "test" if os.path.exists(os.path.join("dataset", "test", "images")) else "val"
        metrics = model.val(
            data=data_yaml_path,
            split=split_target,
            imgsz=exp["imgsz"],
            batch=exp["batch"],
            project="runs/evaluate",
            name=exp["name"],
            exist_ok=True
        )

        eval_records.append({
            "Configuration": exp["name"],
            "Epochs": exp["epochs"],
            "Img_Size": exp["imgsz"],
            "Batch_Size": exp["batch"],
            "Precision": round(float(metrics.box.mp), 4),
            "Recall": round(float(metrics.box.mr), 4),
            "mAP@50": round(float(metrics.box.map50), 4),
            "mAP@50-95": round(float(metrics.box.map), 4)
        })

    if eval_records:
        eval_df = pd.DataFrame(eval_records)
        print("\n" + "=" * 85)
        print("ตารางเปรียบเทียบประสิทธิภาพ (Precision, Recall, mAP) บนชุดข้อมูล Thai BankNotes:")
        print("=" * 85)
        print(eval_df.to_string(index=False))
        eval_df.to_csv("evaluation_comparison.csv", index=False)
        print("\n[SUCCESS] บันทึกผลการประเมินลงในไฟล์ 'evaluation_comparison.csv' เรียบร้อยแล้ว")

if __name__ == "__main__":
    evaluate_models()