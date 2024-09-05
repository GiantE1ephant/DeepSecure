import argparse
import torch
from tool import get_model, load_data, ifgsm_attack, evaluate_model, check_data_loader, calculate_attack_success_rate

def evaluate_with_and_without_attack(model_name, dataset_name, metrics, epsilon=0.1, alpha=0.01, iters=10):
    # 检查 CUDA 设备是否可用，并设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载保存的模型状态
    save_path = "D:/myapps/Pycharm/projectpy/DeepSecure/modelDone/"
    model_path = save_path + f"{model_name}-{dataset_name}.pth"
    model = get_model(model_name, dataset_name).to(device)
    model.load_state_dict(torch.load(model_path))

    # 加载数据集
    train_loader, test_loader = load_data(dataset_name)

    print(f"{dataset_name} dataset has been successfully loaded and is ready for evaluation.")
    print(f"Model '{model_name}' for the '{dataset_name}' dataset has been successfully loaded and is ready for use.")

    # 无攻击评估
    acc_clean, prec_clean, rec_clean, f1_clean = evaluate_model(model, test_loader, device)
    if 'accuracy' in metrics:
        print(f"Clean - Accuracy: {acc_clean:.4f}")
    if 'precision' in metrics:
        print(f"Clean - Precision: {prec_clean:.4f}")
    if 'recall' in metrics:
        print(f"Clean - Recall: {rec_clean:.4f}")
    if 'f1' in metrics:
        print(f"Clean - F1-Score: {f1_clean:.4f}")

    # 有攻击评估
    acc_adv, prec_adv, rec_adv, f1_adv = evaluate_model(model, test_loader, device, ifgsm_attack, epsilon, alpha, iters)

    # 计算攻击成功率
    attack_success_rate = calculate_attack_success_rate(model, test_loader, device, ifgsm_attack, epsilon, alpha, iters)

    if 'accuracy' in metrics:
        print(f"Adversarial - Accuracy: {acc_adv:.4f}")
    if 'precision' in metrics:
        print(f"Adversarial - Precision: {prec_adv:.4f}")
    if 'recall' in metrics:
        print(f"Adversarial - Recall: {rec_adv:.4f}")
    if 'f1' in metrics:
        print(f"Adversarial - F1-Score: {f1_adv:.4f}")

    # 输出扰动阈值和攻击成功率
    print(f"Adversarial Perturbation Threshold (Epsilon): {epsilon}")
    print(f"Adversarial Attack Success Rate: {attack_success_rate:.4f}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate model performance with and without adversarial attacks.")
    parser.add_argument('model_name', type=str, choices=['AlexNet', 'LeNet', 'ResNet'], help='Name of the model to evaluate')
    parser.add_argument('dataset_name', type=str, choices=['CIFAR-10', 'MSTAR'], help='Name of the dataset to use')
    parser.add_argument('--epsilon', type=float, default=0.1, help='Epsilon parameter for IFGSM attack')
    parser.add_argument('--alpha', type=float, default=0.01, help='Alpha parameter for IFGSM attack')
    parser.add_argument('--iters', type=int, default=10, help='Number of iterations for IFGSM attack')

    # 必选参数：指标选择
    parser.add_argument('--metrics', nargs='+', required=True, choices=['accuracy', 'precision', 'recall', 'f1'],
                        help='Choose metrics to display: accuracy, precision, recall, f1 (at least one required)')

    args = parser.parse_args()

    evaluate_with_and_without_attack(args.model_name, args.dataset_name, args.metrics, args.epsilon, args.alpha, args.iters)

if __name__ == "__main__":
    main()
