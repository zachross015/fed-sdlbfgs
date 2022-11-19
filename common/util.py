from models import *
from optims import *



def get_model(model_name):
    if model_name == 'CNN':
        model = CNN(3, len(classes), 32)
    else:
        model = resnet18(len(classes))
    model = model.to(device)
    return model


def get_optim(optim_name):
    if optim_name == 'sdlbfgs':
        return SdLBFGS
    elif optim_name == 'sdlbfgs_layer':
        return SdLBFGSLayer
    elif optim_name == 'adam':
        return optim.Adam
    elif optim_name == 'kfac':
        return KFACOptimizer
    elif optim_name == 'shampoo':
        return Shampoo
    else:
        return optim.SGD
