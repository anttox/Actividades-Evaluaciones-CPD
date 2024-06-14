import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# Configuracion del dispositivo
# Verificamos si hay GPUs disponibles y configuramos el dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_gpus = torch.cuda.device_count()

# Definir el modelo de red neuronal
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        # Primera capa totalmente conectada (entrada: 784, salida: 256)
        self.fc1 = nn.Linear(784, 256)
        # Funcion de activacion ReLU
        self.relu = nn.ReLU()
        # Segunda capa totalmente conectada (entrada: 256, salida: 10)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        # Aplanar las entradas
        x = x.view(x.size(0), -1)
        # Pasar a traves de la primera capa y aplicar ReLU
        x = self.fc1(x)
        x = self.relu(x)
        # Pasar a traves de la segunda capa
        x = self.fc2(x)
        return x

# Cargar el conjunto de datos MNIST
# Definimos las transformaciones para normalizar las imagenes
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

# Iniciamos el modelo y paralelizamos
model = SimpleNN().to(device)
if num_gpus > 1:
    model = nn.DataParallel(model)

# Definimos el optimizador y la funcion de perdida
optimizer = optim.SGD(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

# Entrenar el modelo
for epoch in range(10):
    for data, target in train_loader:
        # Transferimos los datos y etiquetas al dispositivo configurado (GPU)
        data, target = data.to(device), target.to(device)
        
        # Reiniciamos los gradientes del optimizador
        optimizer.zero_grad()
        
        # Forward pass: calculo de las predicciones del modelo
        output = model(data)
        
        # Calculo de la perdida entre las predicciones y las etiquetas
        loss = criterion(output, target)
        
        # Backward pass: calculo de los gradientes
        loss.backward()
        
        # Actualizacion de los parametros del modelo
        optimizer.step()
    
    # Imprimimos la perdida al final de cada epoca
    print(f"Epoca {epoch+1}, Perdida: {loss.item()}")
