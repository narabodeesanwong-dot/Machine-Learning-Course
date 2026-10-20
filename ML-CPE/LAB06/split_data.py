from sklearn.model_selection import train_test_split

def split_dataset(X, y):
    # Split the dataset into training (80%) and testing sets (20%)
    return train_test_split(X, y, test_size=0.2, random_state=42)