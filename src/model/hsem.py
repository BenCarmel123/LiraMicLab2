import torch
import cv2
import sys
from types import ModuleType

# Comprehensive compatibility shim for timm version changes
compatibility_mappings = {
    'timm.layers': 'timm.models.layers',
    'timm.models._efficientnet_blocks': 'timm.models.efficientnet_blocks',
    'timm.models._builder': 'timm.models.builder',
    'timm.models._features': 'timm.models.features',
    'timm.models._hub': 'timm.models.hub',
}

for old_module, new_module in compatibility_mappings.items():
    try:
        parts = new_module.split('.')
        module = __import__(new_module, fromlist=[parts[-1]])
        sys.modules[old_module] = module
    except ImportError:
        sys.modules[old_module] = ModuleType(old_module)

from hsemotion.facial_emotions import HSEmotionRecognizer
import torch.nn as nn

# Monkey-patch to add missing attributes
def patch_efficientnet(model):
    """Add missing attributes for backward compatibility"""
    if hasattr(model, 'conv_stem') and not hasattr(model, 'act1'):
        model.act1 = nn.Identity()
    return model

_original_load = torch.load
def cpu_only_load(*args, **kwargs):
    kwargs["map_location"] = torch.device("cpu")
    kwargs["weights_only"] = False
    result = _original_load(*args, **kwargs)
    
    if hasattr(result, '__class__') and 'EfficientNet' in result.__class__.__name__:
        result = patch_efficientnet(result)
    
    return result

torch.load = cpu_only_load

MODEL_NAME = "enet_b0_8_best_afew"   
_model = None

def load_model():
    global _model
    if _model is None:
        _model = HSEmotionRecognizer(model_name=MODEL_NAME, device="cpu")
        
        if hasattr(_model, 'model'):
            _model.model = patch_efficientnet(_model.model)
    return _model

def get_embedding(image_path):
    """Extract 1280-D facial emotion embedding from an image"""
    model = load_model()
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"Cannot read: {image_path}")

    features = model.extract_features(image)
    return features
