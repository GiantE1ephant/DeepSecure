import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tool import ifgsm_attack, fgsm_attack


# 评估函数
def evaluate_model(model, data_loader, device, attack=None, epsilon=0.1, alpha=0.01, iters=10):
    model.eval()
    y_true = []
    y_pred = []

    for data in data_loader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)

        if attack:
            images = ifgsm_attack(model, images, labels, epsilon, alpha, iters, device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())


    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    return accuracy, precision, recall, f1
