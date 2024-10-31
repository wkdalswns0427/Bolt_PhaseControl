import torch
import numpy as np

# Load the .pt file (replace 'model_5000.pt' with your file path)
model = torch.load('model_5000.pt', map_location=torch.device('cpu'))

# Extract the model state dictionary
model_state_dict = model['model_state_dict']

# Define the mappings for reshaping and filenames
file_mapping = {
    'actor.0.weight': '0_weight.txt',
    'actor.0.bias': '0_bias.txt',
    'actor.2.weight': '2_weight.txt',
    'actor.2.bias': '2_bias.txt',
    'actor.4.weight': '4_weight.txt',
    'actor.4.bias': '4_bias.txt',
    'actor.6.weight': '6_weight.txt',
    'actor.6.bias': '6_bias.txt',
}

reshape_mapping = {
    'actor.0.weight': (512, 120),
    'actor.0.bias': (512, 1),
    'actor.2.weight': (256, 512),
    'actor.2.bias': (256, 1),
    'actor.4.weight': (128, 256),
    'actor.4.bias': (128, 1),
    'actor.6.weight': (6, 128),
    'actor.6.bias': (6, 1),
}

# Save the reshaped components to .txt files
for key, filename in file_mapping.items():
    component = np.reshape(model_state_dict[key].numpy(), reshape_mapping[key])
    np.savetxt(filename, component)

print("Files have been successfully generated!")
