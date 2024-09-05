import torch

def calculate_attack_success_rate(model, test_loader, device, attack_func, epsilon, alpha, iters):
    model.eval()
    total_samples = 0
    successful_attacks = 0

    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        # 原始预测
        with torch.no_grad():
            original_preds = model(images).argmax(dim=1)

        # 应用攻击
        adversarial_images = attack_func(model, images, labels, epsilon, alpha, iters,device)

        # 攻击后的预测
        with torch.no_grad():
            adv_preds = model(adversarial_images).argmax(dim=1)

        # 计算成功攻击的样本
        successful_attacks += (adv_preds != original_preds).sum().item()
        total_samples += labels.size(0)

    # 返回攻击成功率
    return successful_attacks / total_samples
