import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertModel, BertTokenizer

# Configuracion del dispositivo
# Verificamos si hay GPUs disponibles y configuramos el dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_gpus = torch.cuda.device_count()

# Definicion del modelo de Transformer (BERT)
class SimpleBERT(nn.Module):
    def __init__(self):
        super(SimpleBERT, self).__init__()
        # Cargamos un modelo preentrenado de BERT
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        # Capa totalmente conectada para la tarea de clasificación binaria
        self.fc = nn.Linear(768, 2)

    def forward(self, x):
        # Pasamos la entrada por BERT y obtenemos las representaciones
        x = self.bert(x)[0]
        # Utilizamos solo la representacion del token [CLS] para la clasificación
        x = self.fc(x[:, 0, :])
        return x

# Tokenizacion de ejemplo
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
texts = ["Example sentence for BERT model."] * 10
# Tokenizamos las frases de ejemplo, aplicando padding y truncamiento
inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True)

# Iniciamos el modelo y configuracion para multiples GPUs
model = SimpleBERT().to(device)
if num_gpus > 1:
    model = nn.DataParallel(model)

# Configuracion del optimizador y la funcion de perdida
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Entrenamiento del modelo
for epoch in range(10):
    optimizer.zero_grad()
    # Forward pass: calculo de las predicciones del modelo
    outputs = model(inputs['input_ids'].to(device))
    # Calculo de la perdida entre las predicciones y las etiquetas
    loss = criterion(outputs, torch.tensor([1]*10).to(device))
    # Backward pass: calculo de los gradientes
    loss.backward()
    # Actualizacion de los parametros del modelo
    optimizer.step()
    # Imprime la perdida al final de cada epoca
    print(f"Epoca {epoch+1}, Perdida: {loss.item()}")
