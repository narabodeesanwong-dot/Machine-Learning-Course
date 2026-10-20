from sklearn.neural_network import MLPClassifier

def build_model(config_type, epochs):
    if config_type == "Config_1":
        hidden_layers = (16,)
    elif config_type == "Config_2":
        hidden_layers = (32, 16)
        
    model = MLPClassifier(hidden_layer_sizes=hidden_layers, 
                          max_iter=epochs, 
                          random_state=42)
    return model