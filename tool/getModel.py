from model import AlexNet, LeNet, ResNet

# 选择模型
def get_model(model_name,dataset_name):
    if model_name == 'LeNet':
        return LeNet(dataset_name)
    elif model_name == 'AlexNet':
        return AlexNet(dataset_name)
    elif model_name == 'ResNet':
        return ResNet()
    else:
        raise ValueError("Invalid model name")