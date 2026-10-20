# ML-10-Object-Detection-II: Thai BankNotes Object Detection Using YOLO

## 1. วัตถุประสงค์ (Objectives)
1. ประยุกต์ใช้โมเดล YOLO (You Only Look Once) ในการตรวจจับ จำแนกประเภท และระบุตำแหน่งธนบัตรไทยภายในภาพ
2. เปรียบเทียบประสิทธิภาพของโมเดลเมื่อฝึกสอนด้วยจำนวนรอบ (Epochs) และการตั้งค่า (Image Size, Batch Size) ที่แตกต่างกัน
3. ประเมินประสิทธิภาพโมเดลด้วยค่า Precision, Recall และ mAP พร้อมทดสอบตรวจจับวัตถุจากภาพนิ่ง วิดีโอ และกล้องเว็บแคมแบบเรียลไทม์

## 2. รายละเอียดชุดข้อมูลและแหล่งอ้างอิง (Dataset & Reference)
* **ชื่อชุดข้อมูล:** Thai BankNotes Object Detection Dataset (Version 2, 80-20 Split)
* **แหล่งอ้างอิง (Source URL):** https://universe.roboflow.com/dip-lp0qi/thai-banknotes
* **จำนวนข้อมูลทั้งหมด:** 1,000 ภาพ (แบ่งเป็น Train 800 ภาพ, Validation 100 ภาพ, และ Test 100 ภาพ)
* **คลาสที่ตรวจจับ (5 Classes):**
  * `0: THAI100` (ธนบัตร 100 บาท - 161 ภาพในชุด Train / 20.12%)
  * `1: THAI1000` (ธนบัตร 1,000 บาท - 163 ภาพในชุด Train / 20.38%)
  * `2: THAI20` (ธนบัตร 20 บาท - 159 ภาพในชุด Train / 19.88%)
  * `3: THAI50` (ธนบัตร 50 บาท - 157 ภาพในชุด Train / 19.62%)
  * `4: THAI500` (ธนบัตร 500 บาท - 160 ภาพในชุด Train / 20.00%)

## 3. โครงสร้างโปรเจกต์ (Project Structure)
```text
ML-10-Object-Detection-II/
├── dataset/
│   ├── train/{images,labels}/
│   ├── val/{images,labels}/
│   ├── test/{images,labels}/
│   └── data.yaml
├── train.py
├── evaluate.py
├── image_test.py
├── realtime_test.py
├── requirements.txt
└── README.md