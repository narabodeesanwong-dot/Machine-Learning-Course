from ultralytics import YOLO
import cv2
import time
import os

BANKNOTE_VALUES = {
    "THAI20": 20,
    "THAI50": 50,
    "THAI100": 100,
    "THAI500": 500,
    "THAI1000": 1000
}

def run_realtime_detection(camera_index=0):
    weight_path = os.path.join("runs", "detect", "exp2_ep25_img640_b16", "weights", "best.pt")
    if not os.path.exists(weight_path):
        print(f"[WARNING] ไม่พบไฟล์ {weight_path} ระบบจะใช้ yolov8n.pt แทนชั่วคราว")
        weight_path = "yolov8n.pt"

    model = YOLO(weight_path)
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print("ไม่สามารถเปิดกล้องเว็บแคมได้ กรุณาตรวจสอบการเชื่อมต่อกล้อง")
        return

    print("เริ่มการตรวจจับธนบัตรไทยแบบเรียลไทม์... (กดปุ่ม 'q' เพื่อปิดหน้าต่าง)")
    prev_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("สิ้นสุดสัญญาณภาพจากกล้อง")
            break

        # ตรวจจับธนบัตรในเฟรมปัจจุบัน
        results = model.predict(source=frame, conf=0.35, verbose=False)
        annotated_frame = results[0].plot()

        # คำนวณมูลค่าธนบัตรที่ตรวจพบในเฟรมปัจจุบัน
        total_amount = 0
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            total_amount += BANKNOTE_VALUES.get(cls_name, 0)

        # คำนวณค่า FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time > 0 else 0
        prev_time = curr_time

        # แสดงค่า FPS และยอดเงินรวมบนหน้าจอ
        cv2.putText(
            annotated_frame,
            f"FPS: {fps:.1f} | Total: {total_amount} THB",
            (15, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )

        cv2.imshow("Thai BankNotes Real-Time Detection (Press 'q' to Exit)", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_realtime_detection(camera_index=0)