import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Configuracion del dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_gpus = torch.cuda.device_count()

# Modelo de ejemplo
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Creacion de un dataset de ejemplo
x = torch.randn(10000, 784)
y = torch.randint(0, 10, (10000,))
dataset = TensorDataset(x, y)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Division del modelo entre multiples GPUs
model = SimpleNN().to(device)
if num_gpus > 1:
    model = nn.DataParallel(model)

optimizer = optim.SGD(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

# Entrenamiento del modelo
for epoch in range(10):
    for data, target in dataloader:
        # Transferencia de datos a GPU
        data, target = data.to(device), target.to(device)
        
        # Reinicia los gradientes
        optimizer.zero_grad()
        
        # Forward pass
        output = model(data)
        
        # Calculo de la perdida
        loss = criterion(output, target)
        
        # Backward pass (retropropagacion)
        loss.backward()
        
        # Actualizacion de los parametros del modelo
        optimizer.step()
    
    print(f"Epoca {epoch+1}, Perdida: {loss.item()}")
