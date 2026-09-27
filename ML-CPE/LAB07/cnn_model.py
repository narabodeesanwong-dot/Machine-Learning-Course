from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense

def build_cnn(input_shape, num_classes, config_type='basic'):
    model = Sequential()
    
    if config_type == 'basic':
        # Configuration 1: 1 Conv layer, น้อย Neurons
        model.add(Conv1D(filters=16, kernel_size=2, activation='relu', input_shape=input_shape))
        model.add(Flatten())
        model.add(Dense(32, activation='relu'))
        
    elif config_type == 'deep':
        # Configuration 2: 2 Conv layers, มาก Neurons
        model.add(Conv1D(filters=32, kernel_size=2, activation='relu', input_shape=input_shape))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Conv1D(filters=64, kernel_size=2, activation='relu'))
        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
                  
    return model