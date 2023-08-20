from model.DABNet import DABNet
from model.fast_scnn import FastSCNN
from model.CFast_SCNN import CFastSCNN
from model.AEMFast_SCNN import AEMFastSCNN
from model.SBAMFast_SCNN import SBAMFastSCNN


def build_model(model_name, num_classes):
    if model_name == 'DABNet':
        return DABNet(classes=num_classes)

    elif model_name == 'FastSCNN':
        return FastSCNN(num_classes=num_classes)
    
    elif model_name == 'CFastSCNN':
        return CFastSCNN(num_classes=num_classes)
    
    elif model_name == 'AEMFastSCNN':
        return AEMFastSCNN(num_classes=num_classes)
    
    elif model_name == 'SBAMFastSCNN':
        return SBAMFastSCNN(num_classes=num_classes)
    
    
