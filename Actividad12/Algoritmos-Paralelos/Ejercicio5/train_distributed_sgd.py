import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import datasets, transforms

# Configuramos el inicio del proceso para el paralelismo
def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

# Definimos un dataset simple para entrenamiento
class SimpleDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

# Definimos el modelo de red neuronal simple
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Aplanamos el tensor
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def train(rank, world_size, train_loader, model, criterion, optimizer, epochs=5):
    setup(rank, world_size)
    model = model.to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    for epoch in range(epochs):
        ddp_model.train()
        running_loss = 0.0
        for data, target in train_loader:
            data, target = data.to(rank), target.to(rank)

            optimizer.zero_grad()
            outputs = ddp_model(data)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print(f"Rank {rank}, Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")
    
    cleanup()

def main():
    # Cargamos el conjunto de datos MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    
    # Configuramos la division de datos entre los nodos
    world_size = torch.cuda.device_count()
    subset_size = len(dataset) // world_size
    subsets = [Subset(dataset, range(i*subset_size, (i+1)*subset_size)) for i in range(world_size)]
    train_loaders = [DataLoader(subset, batch_size=64, shuffle=True) for subset in subsets]

    model = SimpleNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    mp.spawn(train, args=(world_size, train_loaders[0], model, criterion, optimizer), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()
