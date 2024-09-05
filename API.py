from flask import Flask, request, Response
import torch
import json
from tool import get_model, load_data, ifgsm_attack, evaluate_model, check_data_loader, calculate_attack_success_rate
from collections import OrderedDict

app = Flask(__name__)

@app.route('/evaluate', methods=['GET'])
def evaluate():
    # 获取 GET 请求中的参数
    model_name = request.args.get('model_name')
    dataset_name = request.args.get('dataset_name')
    epsilon = float(request.args.get('epsilon', 0.1))
    alpha = float(request.args.get('alpha', 0.01))
    iters = int(request.args.get('iters', 10))
    metrics = request.args.getlist('metrics')

    # 检查必选参数
    if not model_name or not dataset_name or not metrics:
        return Response(json.dumps({"error": "Missing required parameters: model_name, dataset_name, metrics"}), status=400, mimetype='application/json')

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型
    save_path = "D:/myapps/Pycharm/projectpy/DeepSecure/modelDone/"
    model_path = save_path + f"{model_name}-{dataset_name}.pth"
    model = get_model(model_name, dataset_name).to(device)
    model.load_state_dict(torch.load(model_path))

    # 加载数据集
    train_loader, test_loader = load_data(dataset_name)

    # 无攻击评估
    acc_clean, prec_clean, rec_clean, f1_clean = evaluate_model(model, test_loader, device)

    # 有攻击评估
    acc_adv, prec_adv, rec_adv, f1_adv = evaluate_model(model, test_loader, device, ifgsm_attack, epsilon, alpha, iters)

    # 计算攻击成功率
    attack_success_rate = calculate_attack_success_rate(model, test_loader, device, ifgsm_attack, epsilon, alpha, iters)

    # 使用 OrderedDict 确保字段顺序
    result = OrderedDict([
        ("clean", {
            "accuracy": acc_clean if 'accuracy' in metrics else None,
            "precision": prec_clean if 'precision' in metrics else None,
            "recall": rec_clean if 'recall' in metrics else None,
            "f1": f1_clean if 'f1' in metrics else None,
        }),
        ("adversarial", {
            "accuracy": acc_adv if 'accuracy' in metrics else None,
            "precision": prec_adv if 'precision' in metrics else None,
            "recall": rec_adv if 'recall' in metrics else None,
            "f1": f1_adv if 'f1' in metrics else None,
        }),
        ("attack_success_rate", attack_success_rate),
        ("Adversarial Perturbation Threshold (Epsilon)", epsilon)
    ])

    # 使用 json.dumps() 手动构建 JSON 响应
    response = json.dumps(result, ensure_ascii=False)
    return Response(response, mimetype='application/json')

if __name__ == '__main__':
    app.run(debug=True)
