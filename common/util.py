from models import CNN, resnet18
from optims import SdLBFGS, SdLBFGSLayer, KFACOptimizer, Shampoo
import torch.optim as optim


def get_model(model_name):
    if model_name == 'CNN':
        model = CNN(3, 10, 32)
    else:
        model = resnet18(10)
    return model


def get_optim(optim_name, model, **kwargs):
    if optim_name == 'sdlbfgs':
        return SdLBFGS(model.parameters(), **kwargs)
    elif optim_name == 'sdlbfgs_layer':
        return SdLBFGSLayer(model.parameters(), **kwargs)
    elif optim_name == 'adam':
        return optim.Adam(model.parameters())
    elif optim_name == 'kfac':
        return KFACOptimizer(model, lr=0.0001)
    elif optim_name == 'shampoo':
        return Shampoo(model.parameters())
    else:
        return optim.SGD(model.parameters(), lr=0.005)
