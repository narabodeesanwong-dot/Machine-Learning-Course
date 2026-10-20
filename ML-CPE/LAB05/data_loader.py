import pandas as pd

def load_data(filepath='winequality-red.csv'):
    # รองรับ delimiter ทั้ง comma (,) และ semicolon (;) เพื่อความยืดหยุ่นในการอ่านไฟล์
    try:
        # ลองอ่านไฟล์โดยใช้ ; เป็นตัวคั่นก่อน (ไฟล์ไวน์บางเวอร์ชันใช้ ;)
        df = pd.pd.read_csv(filepath, sep=';')
        
        # ถ้าอ่านแล้วข้อมูลรวมเป็นคอลัมน์เดียว แปลว่าไฟล์จริงอาจใช้ , เป็นตัวคั่น
        if df.shape[1] == 1:
            df = pd.read_csv(filepath, sep=',')
    except Exception:
        # ดักจับ Error กรณีอ่านพัง ให้บังคับอ่านด้วย , เป็นตัวคั่น
        df = pd.read_csv(filepath, sep=',')
        
    return df # ส่งคืนข้อมูลในรูปแบบ DataFrame
