import torch
import torch.nn as nn

def ifgsm_attack(model, x, y, epsilon, alpha, num_iter, device):
    model.eval()  # 确保模型处于评估模式
    x = x.to(device)
    y = y.to(device)

    x_adv = x.clone().detach().requires_grad_(True)  # 确保 x_adv 需要梯度

    loss_fn = nn.CrossEntropyLoss()

    for _ in range(num_iter):
        # 计算损失
        outputs = model(x_adv)
        loss = loss_fn(outputs, y)

        # 清除之前的梯度
        model.zero_grad()
        # 计算梯度
        loss.backward(retain_graph=True)

        # 获取梯度并更新对抗样本
        grad = x_adv.grad.detach()  # 获取梯度
        x_adv = x_adv + alpha * grad.sign()  # 更新对抗样本

        # 确保对抗样本在合理范围内
        x_adv = torch.clamp(x_adv, min=0, max=1)  # 确保对抗样本在[0,1]范围内
        x_adv = torch.max(torch.min(x_adv, x + epsilon), x - epsilon)  # 应用epsilon限制

        # 重新设置 requires_grad
        x_adv = x_adv.detach().requires_grad_(True)

    return x_adv

def fgsm_attack(model, x, y, epsilon, device):
    model.eval()  # 确保模型处于评估模式
    x = x.to(device)
    y = y.to(device)

    # 计算损失函数
    loss_fn = nn.CrossEntropyLoss()

    # 计算对抗样本
    x.requires_grad = True
    outputs = model(x)
    loss = loss_fn(outputs, y)
    model.zero_grad()
    loss.backward()

    # 获取梯度并计算对抗样本
    grad = x.grad.data
    x_adv = x + epsilon * grad.sign()

    # 确保对抗样本在合理范围内
    x_adv = torch.clamp(x_adv, 0, 1)

    return x_adv
