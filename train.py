import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

from tool import get_model, load_data, check_data_loader

def evaluate_model(model, data_loader, device):
    model.eval()
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    return all_labels, all_preds

def train_and_evaluate(model_name, dataset_name, model_path, num_epochs=5, learning_rate=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载数据集
    train_loader, test_loader = load_data(dataset_name)
    check_data_loader(train_loader)

    # 初始化模型
    model = get_model(model_name, dataset_name).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Evaluation on the test set
        all_labels, all_preds = evaluate_model(model, test_loader, device)
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='weighted')
        recall = recall_score(all_labels, all_preds, average='weighted')
        f1 = f1_score(all_labels, all_preds, average='weighted')

        print(f"Epoch {epoch + 1}: Loss: {running_loss / len(train_loader):.4f}, "
              f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, "
              f"Recall: {recall:.4f}, F1-score: {f1:.4f}")

    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

def run_model_training(model_name, dataset_name):
    save_path = "D:/myapps/Pycharm/projectpy/DeepSecure/modelDone/"
    model_path = save_path + f"{model_name}-{dataset_name}.pth"
    train_and_evaluate(model_name, dataset_name, model_path)

def main():
    parser = argparse.ArgumentParser(description="Train and evaluate a model.")
    parser.add_argument('model_name', type=str, choices=['AlexNet', 'LeNet', 'ResNet'], help='Name of the model to train and evaluate')
    parser.add_argument('dataset_name', type=str, choices=['CIFAR-10', 'MSTAR'], help='Name of the dataset to use')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs for training')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for training')

    args = parser.parse_args()

    run_model_training(args.model_name, args.dataset_name)

if __name__ == "__main__":
    main()
