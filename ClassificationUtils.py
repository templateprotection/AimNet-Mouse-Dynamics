import torch
from matplotlib import pyplot as plt
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm


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

