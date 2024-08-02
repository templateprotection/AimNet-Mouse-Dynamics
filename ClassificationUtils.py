import torch
from matplotlib import pyplot as plt
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

def create_data_loader(sequences, labels, batch_size=32, shuffle=True):
    sequences_tensor = torch.tensor(sequences, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    dataset = TensorDataset(sequences_tensor, labels_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader


def train_classification_model(model, train_loader, validation_loader, model_output, num_epochs=100, learning_rate=0.001):
    best_val_acc = 0.0
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    train_accs = []
    val_accs = []
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0
        for (inputs, labels) in tqdm(train_loader, total=len(train_loader)):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            tc = (predicted == labels).sum().item()
            tt = labels.size(0)
            train_correct += tc
            train_total += tt
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_accs.append(train_correct / train_total)

        model.eval()
        valid_correct = 0
        valid_total = 0
        with torch.no_grad():
            for inputs, labels in validation_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                valid_total += labels.size(0)
                valid_correct += (predicted == labels).sum().item()

        val_acc = valid_correct / valid_total
        val_accs.append(val_acc)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Acc: {100 * train_correct / train_total:.2f}%, Valid Acc: {100 * valid_correct / valid_total:.2f}%')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"./models/{model_output}")

    print('Training complete')

    plt.plot(train_accs)
    plt.plot(val_accs)
    plt.legend(['Training Accuracy', 'Validation Accuracy'])
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.show()


class LSTMClassifier(nn.Module):
    def __init__(self, input_size, output_size):
        super(LSTMClassifier, self).__init__()

        self.batch_norm1 = nn.BatchNorm1d(input_size)
        self.lstm1 = nn.LSTM(input_size, 128, batch_first=True)
        self.dropout = nn.Dropout(0.5)
        self.batch_norm2 = nn.BatchNorm1d(128)
        self.lstm2 = nn.LSTM(128, 128, batch_first=True)
        self.fc = nn.Linear(128, output_size)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.batch_norm1(x)
        x = x.transpose(1, 2)
        x, _ = self.lstm1(x)
        x = self.dropout(x)
        x = x.transpose(1, 2)
        x = self.batch_norm2(x)
        x = x.transpose(1, 2)
        x, _ = self.lstm2(x)
        x = x[:, -1, :]
        x = self.fc(x)
        return x


class GRUClassifier(nn.Module):
    def __init__(self, input_size, output_size):
        super(GRUClassifier, self).__init__()

        self.batch_norm1 = nn.BatchNorm1d(input_size)
        self.lstm1 = nn.LSTM(input_size, 128, batch_first=True)
        self.dropout = nn.Dropout(0.5)
        self.batch_norm2 = nn.BatchNorm1d(128)
        self.lstm2 = nn.LSTM(128, 128, batch_first=True)
        self.fc = nn.Linear(128, output_size)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.batch_norm1(x)
        x = x.transpose(1, 2)
        x, _ = self.lstm1(x)
        x = self.dropout(x)
        x = x.transpose(1, 2)
        x = self.batch_norm2(x)
        x = x.transpose(1, 2)
        x, _ = self.lstm2(x)
        x = x[:, -1, :]
        x = self.fc(x)
        return x


class CNNClassifier(nn.Module):
    def __init__(self, input_size, output_size, seq_length):
        super(CNNClassifier, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=64, kernel_size=3,
                               padding=1)
        self.batch_norm1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3,
                               padding=1)
        self.batch_norm2 = nn.BatchNorm1d(128)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(128 * (seq_length // 2), 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, output_size)

    def forward(self, x):
        x = x.transpose(1, 2)  # (batch_size, input_size, SEQ_LENGTH)
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = nn.ReLU()(x)
        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = nn.ReLU()(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, transformer_emb_dim=64, num_heads=4, num_layers=2, dropout=0.1):
        super(TransformerClassifier, self).__init__()

        self.positional_encoding = nn.Parameter(torch.randn(1, 100, transformer_emb_dim))
        self.embedding = nn.Linear(input_dim, transformer_emb_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=transformer_emb_dim, nhead=num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(transformer_emb_dim, num_classes)

    def forward(self, x):
        batch_size, seq_length, _ = x.size()
        x = self.embedding(x) + self.positional_encoding[:, :seq_length, :]
        x = x.transpose(0, 1)
        x = self.transformer_encoder(x)
        x = x[-1, :, :]  # (batch_size, embed_dim)
        x = self.dropout(x)
        x = self.fc(x)

        return x
