import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import time

# Configuracion de la inicializacion del proceso para el paralelismo
def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

# Definimos un dataset simple para entrenamiento
class SimpleDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

# Definimos la arquitectura del transformer simple
class SimpleTransformer(nn.Module):
    def __init__(self):
        super(SimpleTransformer, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.fc = nn.Linear(self.bert.config.hidden_size, 2)  # Asumiendo una tarea de clasificación binaria

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs[1]
        return self.fc(cls_output)

def train(rank, world_size, train_loader, model, criterion, optimizer, epochs=3):
    setup(rank, world_size)
    model = model.to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    for epoch in range(epochs):
        ddp_model.train()
        running_loss = 0.0
        for batch in train_loader:
            input_ids = batch['input_ids'].to(rank)
            attention_mask = batch['attention_mask'].to(rank)
            labels = batch['label'].to(rank)

            optimizer.zero_grad()
            outputs = ddp_model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print(f"Rank {rank}, Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")
    
    cleanup()

def main():
    # Definimos los textos y etiquetas de ejemplo
    texts = ["This is a positive example.", "This is a negative example."] * 100
    labels = [1, 0] * 100

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    dataset = SimpleDataset(texts, labels, tokenizer)
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = SimpleTransformer()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=2e-5)

    world_size = torch.cuda.device_count()
    mp.spawn(train, args=(world_size, train_loader, model, criterion, optimizer), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()
