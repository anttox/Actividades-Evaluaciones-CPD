import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# Configuracion del dispositivo
# Verificamos si hay GPUs disponibles y configuramos el dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_gpus = torch.cuda.device_count()

# Definicion del modelo de red neuronal convolucional (CNN)
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Primera capa convolucional (entrada: 1 canal, salida: 32 canales, kernel de 3x3)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        # Segunda capa convolucional (entrada: 32 canales, salida: 64 canales, kernel de 3x3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        # Primera capa totalmente conectada (entrada: 64*24*24, salida: 128)
        self.fc1 = nn.Linear(64*24*24, 128)
        # Capa de salida (entrada: 128, salida: 10)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # Aplicamos la primera convolucion y la funcion de activación ReLU
        x = self.conv1(x)
        x = nn.functional.relu(x)
        # Aplicamos la segunda convolucion y la funcion de activacion ReLU
        x = self.conv2(x)
        x = nn.functional.relu(x)
        # Aplanamos los datos para las capas totalmente conectadas
        x = x.view(x.size(0), -1)  # Aseguramos que el tamaño del batch este correcto
        # Aplicamos la primera capa totalmente conectada
        x = self.fc1(x)
        x = nn.functional.relu(x)
        # Aplicamos la capa de salida
        x = self.fc2(x)
        return x

# Carga de datos de ejemplo (MNIST)
# Definimos las transformaciones para normalizar las imagenes
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Descargamos y preparamos el conjunto de datos de entrenamiento
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

# Iniciamos el modelo y configuracion para multiples GPUs
model = SimpleCNN().to(device)
if num_gpus > 1:
    model = nn.DataParallel(model)

# Configuracion del optimizador y la funcion de perdida
optimizer = optim.SGD(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

# Entrenamiento del modelo
for epoch in range(10):
    for data, target in train_loader:
        # Transferencia de datos y etiquetas a la GPU
        data, target = data.to(device), target.to(device)
        
        # Reinicia los gradientes del optimizador
        optimizer.zero_grad()
        
        # Forward pass: calculo de las predicciones del modelo
        output = model(data)
        
        # Calculo de la perdida entre las predicciones y las etiquetas
        loss = criterion(output, target)
        
        # Backward pass: calculo de los gradientes
        loss.backward()
        
        # Actualizacion de los parametros del modelo
        optimizer.step()
    
    # Imprime la perdida al final de cada epoca
    print(f"Epoca {epoch+1}, Perdida: {loss.item()}")

