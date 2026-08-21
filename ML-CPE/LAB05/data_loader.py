import pandas as pd

def load_data(filepath='winequality-red.csv'):
    # รองรับ delimiter ทั้ง comma (,) และ semicolon (;)
    try:
        df = pd.read_csv(filepath, sep=';')
        if df.shape[1] == 1:
            df = pd.read_csv(filepath, sep=',')
    except Exception:
        df = pd.read_csv(filepath, sep=',')
    return df