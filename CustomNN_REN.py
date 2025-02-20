import torch
import torch.nn as nn
from REN import RENLayer
from utils import uniform

class CustomNN(nn.Module):
    def __init__(self, one_time_input_length, intermediate_output_length, control_input_shape, lstm_hidden_size, num_lstm_layers):
        super(CustomNN, self).__init__()
        self.linear_density = 5
        # First part: Linear layers
        self.linear_part = nn.Sequential(
            nn.Linear(one_time_input_length, self.linear_density ).to(torch.float32),
            nn.ReLU(),
            nn.Linear(self.linear_density , self.linear_density ).to(torch.float32),
            nn.ReLU(),
            nn.Linear(self.linear_density , self.linear_density ).to(torch.float32),
            nn.ReLU(),
            nn.Linear(self.linear_density , self.linear_density ).to(torch.float32),
            nn.ReLU(),
            nn.Linear(self.linear_density , self.linear_density ).to(torch.float32),
            nn.ReLU(),
            nn.Linear(self.linear_density , self.linear_density).to(torch.float32),
            nn.ReLU(),
            nn.Linear(self.linear_density , self.linear_density ).to(torch.float32),
            nn.ReLU(),
            nn.Linear(self.linear_density , self.linear_density ).to(torch.float32),
            nn.ReLU(),
            nn.Linear(self.linear_density , self.linear_density ).to(torch.float32),
            nn.ReLU(),
            nn.Linear(self.linear_density , self.linear_density ).to(torch.float32),
            nn.ReLU(),
            nn.Linear(self.linear_density , self.linear_density ).to(torch.float32),
            nn.ReLU(),
            nn.Linear(self.linear_density , self.linear_density ).to(torch.float32),
            nn.ReLU(),
            nn.Linear(self.linear_density , intermediate_output_length ).to(torch.float32),
            nn.ReLU()
        )
        
        # Second part: LSTM layers
        self.lstm = RENLayer(1,3,1,5,3)
        self.x_next = uniform(1,3)
       
        
    def forward(self, one_time_input, control_inputs, bin_level):
        control_inputs = control_inputs.to(torch.float32)
        one_time_input = one_time_input.to(torch.float32)
        # Process the one-time input through the linear part
        intermediate_output = self.linear_part(one_time_input)

        # Prepare the LSTM input by combining intermediate output with control inputs
        
        combined_input = torch.cat((intermediate_output.squeeze(), control_inputs.squeeze(), bin_level), dim=0)
        _, bin_level = self.lstm(self.x_next,combined_input.unsqueeze(0).to(torch.float32))
        
        return bin_level

'''# Parameters (example values)
ONE_TIME_INPUT_LENGTH = 10
INTERMEDIATE_OUTPUT_LENGTH = 16
CONTROL_INPUT_SHAPE = (5, 8)  # N_STEPS is assumed to be 5 here
LSTM_HIDDEN_SIZE = 32
NUM_LSTM_LAYERS = 2
N_STEPS = 5

# Initialize the model
model = CustomNN(ONE_TIME_INPUT_LENGTH, INTERMEDIATE_OUTPUT_LENGTH, CONTROL_INPUT_SHAPE, LSTM_HIDDEN_SIZE, NUM_LSTM_LAYERS)

# Example input tensors
one_time_input = torch.randn(1, ONE_TIME_INPUT_LENGTH)
control_inputs = torch.randn(1, N_STEPS, CONTROL_INPUT_SHAPE[1])

# Forward pass
bin_level_output = model(one_time_input, control_inputs)
print(bin_level_output)
'''