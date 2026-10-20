from ultralytics import YOLO
import cv2
import glob
import os

# แปลงชื่อคลาสเป็นมูลค่าเงินบาทเพื่อสรุปผลใน Console
BANKNOTE_VALUES = {
    "THAI20": 20,
    "THAI50": 50,
    "THAI100": 100,
    "THAI500": 500,
    "THAI1000": 1000
}

def test_images_and_video():
    weight_path = os.path.join("runs", "detect", "exp2_ep25_img640_b16", "weights", "best.pt")
    if not os.path.exists(weight_path):
        print(f"[WARNING] ไม่พบไฟล์ {weight_path} ระบบจะใช้ yolov8n.pt เริ่มต้นแทนชั่วคราว")
        weight_path = "yolov8n.pt"

    model = YOLO(weight_path)
    output_dir = os.path.join("outputs", "image_test")
    os.makedirs(output_dir, exist_ok=True)

    # ดึงภาพจาก dataset/test/images (หรือ dataset/val/images) มาทดสอบ 5 ภาพ
    test_images = glob.glob(os.path.join("dataset", "test", "images", "*.*"))
    if not test_images:
        test_images = glob.glob(os.path.join("dataset", "val", "images", "*.*"))
    test_images = test_images[:5]

    if not test_images:
        print("ไม่พบไฟล์รูปภาพใน dataset/test/images หรือ dataset/val/images")
    else:
        print("=" * 70)
        print("ผลการตรวจจับธนบัตรไทยจากภาพนิ่ง (Image Object Detection Results):")
        print("=" * 70)
        for img_path in test_images:
            results = model.predict(source=img_path, conf=0.25, verbose=False)
            for r in results:
                annotated_img = r.plot()
                filename = os.path.basename(img_path)
                save_path = os.path.join(output_dir, f"pred_{filename}")
                cv2.imwrite(save_path, annotated_img)

                total_baht = 0
                print(f"\nไฟล์ภาพ: {filename} -> บันทึกผลลัพธ์ที่: {save_path}")
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    cls_name = model.names[cls_id]
                    conf = float(box.conf[0])
                    coords = [round(x, 1) for x in box.xyxy[0].tolist()]
                    total_baht += BANKNOTE_VALUES.get(cls_name, 0)
                    print(f"  - คลาส: {cls_name:<8} | Confidence: {conf:.2f} | Bounding Box: {coords}")
                print(f"  >> มูลค่าธนบัตรรวมในภาพ: {total_baht:,} บาท")

    # ทดสอบกับไฟล์วิดีโอ (หากมีไฟล์ sample_video.mp4 ในโฟลเดอร์โปรเจกต์)
    video_path = "sample_video.mp4"
    if os.path.exists(video_path):
        print(f"\nกำลังตรวจจับธนบัตรในไฟล์วิดีโอ: {video_path} ...")
        model.predict(
            source=video_path,
            conf=0.25,
            save=True,
            project="outputs",
            name="video_test",
            exist_ok=True
        )
        print("[SUCCESS] บันทึกวิดีโอผลลัพธ์ไว้ใน 'outputs/video_test' เรียบร้อยแล้ว")

if __name__ == "__main__":
    test_images_and_video()