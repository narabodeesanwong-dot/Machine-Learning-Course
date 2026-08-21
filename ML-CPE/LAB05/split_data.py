import numpy as np
import json
from sklearn.model_selection import train_test_split

def prepare_and_split(df, target_col='quality', test_size=0.2, random_state=42, output_dir='outputs'):
    X = df.drop(columns=[target_col]).values
    y = df[target_col].values
    
    # บันทึก Features และ Labels ทั้งหมด
    np.save(f'{output_dir}/features.npy', X)
    np.save(f'{output_dir}/labels.npy', y)
    
    classes = [int(c) for c in np.unique(y)]
    with open(f'{output_dir}/classes.json', 'w') as f:
        json.dump(classes, f)
        
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    np.save(f'{output_dir}/X_train.npy', X_train)
    np.save(f'{output_dir}/X_test.npy', X_test)
    np.save(f'{output_dir}/y_train.npy', y_train)
    np.save(f'{output_dir}/y_test.npy', y_test)
    
    return X_train, X_test, y_train, y_test