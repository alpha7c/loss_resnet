from lop.nets.deep_ffnn import DeepFFNN
from lop.nets.linear import MyLinear

def build_model(agent_type, input_size, num_outputs, num_features, num_hidden_layers):
    
    if agent_type == 'linear':
        return MyLinear(input_size=input_size, num_outputs=num_outputs)
    else:
        return DeepFFNN(
            input_size=input_size,
            num_features=num_features,
            num_outputs=num_outputs,
            num_hidden_layers=num_hidden_layers
        )