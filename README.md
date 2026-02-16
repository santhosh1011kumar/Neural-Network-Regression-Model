# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

This code builds and trains a feedforward neural network in PyTorch for a regression task.
The model takes a single input feature, passes it through two hidden layers with ReLU activation, and predicts one continuous output.
It uses MSE loss and RMSProp optimizer to minimize the error between predictions and actual values over training epochs.

## Neural Network Model

<img width="1073" height="691" alt="image" src="https://github.com/user-attachments/assets/0c9a3477-784c-4316-baef-11c35368797d" />

## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name:SANTHOSH KUMAR A
### Register Number: 212224230250
```python
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


dataset = pd.read_excel(r"/content/dataset 1.xlsx", header=None)
dataset.columns = ['INPUT', 'OUTPUT']

X = dataset[['INPUT']].values
y = dataset[['OUTPUT']].values


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=33
)


scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(1, 8)
        self.fc2 = nn.Linear(8, 10)
        self.fc3 = nn.Linear(10, 1)

        self.relu = nn.ReLU()
        self.history = {'loss': []}

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


ai_brain = NeuralNet()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(ai_brain.parameters(), lr=0.001)



def train_model(model, X_train, y_train, criterion, optimizer, epochs=2000):

    for epoch in range(epochs):
        optimizer.zero_grad()

        output = model(X_train)
        loss = criterion(output, y_train)

        loss.backward()
        optimizer.step()

        model.history['loss'].append(loss.item())

        if epoch % 200 == 0:
            print(f"Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}")



train_model(ai_brain, X_train_tensor, y_train_tensor, criterion, optimizer)



with torch.no_grad():
    test_loss = criterion(ai_brain(X_test_tensor), y_test_tensor)
    print(f"Test Loss: {test_loss.item():.6f}")



loss_df = pd.DataFrame(ai_brain.history)

loss_df.plot()
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss during Training")
plt.show()


X_new = torch.tensor([[9]], dtype=torch.float32)

X_new_scaled = scaler.transform(X_new)
X_new_tensor = torch.tensor(X_new_scaled, dtype=torch.float32)

prediction = ai_brain(X_new_tensor).item()

print(f"Prediction: {prediction}")

```
## Dataset Information

<img width="280" height="476" alt="image" src="https://github.com/user-attachments/assets/698055ed-1bd3-403f-9964-aaee5f9502ae" />

## OUTPUT
<img width="511" height="249" alt="image" src="https://github.com/user-attachments/assets/9b07d079-4c2b-4270-a615-7e0e651613b1" />

### Training Loss Vs Iteration Plot

<img width="821" height="602" alt="image" src="https://github.com/user-attachments/assets/bf5ec4fa-3cd8-4fa4-931a-918cc65805a4" />

### New Sample Data Prediction
```
X_n1_1 = torch.tensor([[9]], dtype=torch.float32)
prediction = santho(torch.tensor(scaler.transform(X_n1_1), dtype=torch.float32)).item()
print(f'Prediction: {prediction}')
```
<img width="912" height="52" alt="image" src="https://github.com/user-attachments/assets/6e904fcc-409c-43bf-ae28-064e3c41a6d5" />

## RESULT

Successfully executed the code to develop a neural network regression model.
