import torch
import torch.nn as nn
import torch.optim as optim

import torch
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, traffic, real_tons, minerals, bin_v, feeder_v, labels):
        # real_tons is a 1D tensor of size (num_samples,)
        # minerals, bin_v, and feeder_v are 2D tensors of size (num_samples, n), (num_samples, m), and (num_samples, z)
        # labels is a tensor of size (num_samples, 1) for regression
        self.traffic =traffic
        self.real_tons = real_tons
        self.minerals = minerals
        self.bin_v = bin_v
        self.feeder_v = feeder_v
        self.labels = labels

    def __len__(self):
        return len(self.real_tons)

    def __getitem__(self, idx):
        traffic = self.traffic[idx]
        real_tons_sample = self.real_tons[idx]
        minerals_sample = self.minerals[idx]
        bin_v_sample = self.bin_v[idx]
        feeder_v_sample = self.feeder_v[idx]
        label = self.labels[idx]
        
        # Return a tuple (inputs, labels), where inputs is the combined input tensor
        inputs = (traffic, real_tons_sample, minerals_sample, bin_v_sample, feeder_v_sample)
        return inputs, label




class Network(nn.Module):
    def __init__(self, mineral_size, sequence_length, batch_size):
        super(Network, self).__init__()
        self.mineral_size = mineral_size
        self.sequence_length = sequence_length
        self.batch_size = batch_size

        self.mineral_parameter = nn.Linear(mineral_size, 1).to(torch.float32)
        self.p_ar_bin = nn.Linear(sequence_length, 2).to(torch.float32)
        self.p_ar_feed = nn.Linear(sequence_length, 2).to(torch.float32)
        self.p_ar_traffic = nn.Linear(sequence_length, 2).to(torch.float32)

        self.Linear_intermediate = nn.Linear(7, 1).to(torch.float32)
        self.nonlin = nn.Tanh()
        
        # Proper initialization with requires_grad handling
        self.c = torch.ones((self.batch_size,), requires_grad=True)  # Replace 'cpu' with 'cuda' if using GPU

    def forward(self, inputs):
        traffic_light , real_tons, minerals, bin_v, feeder_v = inputs
        p = self.mineral_parameter(minerals.to(torch.float32))

        self.c = self.c * (1.1 + p).squeeze(1) + real_tons
        
        ar_bin = self.p_ar_bin(bin_v.to(torch.float32).squeeze(-1))
        ar_feed = self.p_ar_feed(feeder_v.to(torch.float32).squeeze(-1))

        ar_traffic = self.p_ar_traffic(traffic_light.to(torch.float32).squeeze(-1))

        # Assuming ar_bin and ar_feed are of compatible shapes
        x = torch.cat([ar_traffic, ar_bin, ar_feed, self.c.unsqueeze(1)], dim=1)  # Concatenate along the last dimension
        
        
        x= self.Linear_intermediate(x.to(torch.float32))
        x = self.nonlin(x)

        return x
    
    
    def train_model(self, data_loader, num_epochs=10, learning_rate=0.01):
        # Define the loss function (e.g., MSELoss for regression)
        criterion = nn.MSELoss()  # Use another loss function for classification if needed

        # Define the optimizer (Adam in this case)
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        # Loop through the epochs
        for epoch in range(num_epochs):
            running_loss = 0.0

            # Iterate over the data loader
            for inputs, labels in data_loader:
                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = self(inputs).to(torch.float32)

                # Calculate loss
                loss = criterion(outputs.squeeze(-1), labels.to(torch.float32)).to(torch.float32)

                # Backward pass (compute gradients)
                loss.backward(retain_graph=True)

                # Update the weights
                optimizer.step()

                # Print statistics
                running_loss = loss.item()

                # Print the average loss for this epoch
                print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss }")

if __name__=="__main__":

    sequence_length = 40
    mineral_size =10
    batch_size=100
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np


    df=pd.read_csv("merged.csv")

    df_model = df[['Bin Level', 'traffic_light', 'Real Tons', 'Bond Work Index',
        'Deep Work Index', 'feeder 1st Motor']]

    mineral_columns = ['Copper Grade', 'Py', 'Iron', 'Arsenic', 'Mo', 'Kao', 'Piro', 'Bn', 'Mus', 'Sulfide']



    df = pd.concat([df_model, df[mineral_columns]], axis=1)
    columns_to_replace = ['Real Tons', 'Bond Work Index', 'Deep Work Index','feeder 1st Motor','Copper Grade', 'Py', 'Iron', 'Arsenic', 'Mo', 'Kao', 'Piro', 'Bn', 'Mus', 'Sulfide']
    df[columns_to_replace] = df[columns_to_replace].mask(df[columns_to_replace] == 0).ffill()


    def remove_outliers_mean_based(df, deviation_factor=4.0):
        for column in df.select_dtypes(include=[np.number]).columns:
            mean = df[column].mean()
            std_dev = df[column].std()
            lower_bound = mean - deviation_factor * std_dev
            upper_bound = mean + deviation_factor * std_dev
            df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
        return df

    # Define a function to remove outliers based on min-max range with a tolerance
    def remove_outliers_minmax_based(df, tolerance_factor=0.1):
        for column in df.select_dtypes(include=[np.number]).columns:
            min_value = df[column].min()
            max_value = df[column].max()
            range_tolerance = (max_value - min_value) * tolerance_factor
            lower_bound = min_value - range_tolerance
            upper_bound = max_value + range_tolerance
            df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
        return df

    # Choose the method: 'mean' or 'minmax'
    method = 'mean'  # Change to 'minmax' if you want to use the min-max based method

    deviation_factor = 4.0# For mean-based method  this threshold can be changed
    minmax_tolerance = 0.1  # For min-max based method #DONT USE 

    if method == 'mean':
        cleaned_df = remove_outliers_mean_based(df, deviation_factor=deviation_factor)
    elif method == 'minmax':
        cleaned_df = remove_outliers_minmax_based(df, tolerance_factor=minmax_tolerance)

    #print the new data frame 
    df= cleaned_df
    X = df[['traffic_light', 'Real Tons', 'Bond Work Index', 'Bin Level', 'Deep Work Index', 'feeder 1st Motor', 'Copper Grade', 'Py', 'Iron', 'Arsenic', 'Mo', 'Kao', 'Piro', 'Bn', 'Mus', 'Sulfide']]

    y = df['Bin Level']

    X = df[['traffic_light', 'Real Tons',  'Bin Level',  'feeder 1st Motor', 'Copper Grade', 'Py', 'Iron', 'Arsenic', 'Mo', 'Kao', 'Piro', 'Bn', 'Mus', 'Sulfide']]

    minerals=X[['Copper Grade', 'Py', 'Iron', 'Arsenic', 'Mo', 'Kao', 'Piro', 'Bn', 'Mus', 'Sulfide']]
    real_tons=X[['Real Tons']]
    feeder=X[['feeder 1st Motor']]
    traffic_light=X[['traffic_light']]
    bin_level=X[['Bin Level']]


    minerals = torch.tensor(X[['Copper Grade', 'Py', 'Iron', 'Arsenic', 'Mo', 'Kao', 'Piro', 'Bn', 'Mus', 'Sulfide']].values)
    real_tons = torch.tensor(X[['Real Tons']].values)
    feeder = torch.tensor(X[['feeder 1st Motor']].values)
    traffic_light = torch.tensor(X[['traffic_light']].values)
    bin_level= torch.tensor(X[['Bin Level']].values)

    tensors_list = traffic_light,real_tons, minerals, bin_level, feeder 

    y = torch.tensor(df['Bin Level'].values, dtype=torch.float32)
   
    batch_size =100
    sequence_length=40
    mineral_size=10

    minerals_v = []
    real_tons_v = []
    feeder_v = []
    traffic_light_v =[]
    bin_level_v = []
    y_v = []
    for i in range(sequence_length, 100000+sequence_length):
        minerals_v.append(minerals[i].unsqueeze(0))
        real_tons_v.append(real_tons[i])
        feeder_v.append(feeder[i-sequence_length:i,:].unsqueeze(0))
        traffic_light_v.append(traffic_light[i-sequence_length:i,:].unsqueeze(0))
        bin_level_v.append(bin_level[i-sequence_length:i, :].unsqueeze(0))
        y_v.append(bin_level[i+1])
    
    minerals =torch.cat(minerals_v, dim=0)
    real_tons = torch.cat(real_tons_v, dim=0)
    feeder = torch.cat(feeder_v, dim=0)
    traffic = torch.cat(traffic_light_v, dim=0)
    bin_level = torch.cat(bin_level_v, dim=0)
    y = torch.cat(y_v, dim=0)

    # Create the dataset and DataLoader
    dataset = CustomDataset(traffic, real_tons, minerals, bin_level, feeder, y)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Instantiate the model
    n = Network(mineral_size=mineral_size, sequence_length=sequence_length, batch_size=batch_size)

    # Train the model
    n.train_model(data_loader, num_epochs=100, learning_rate=0.01)