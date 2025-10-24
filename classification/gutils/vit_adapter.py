from torch import nn
from pytorch_pretrained_vit import ViT as ViT_pretrained


def get_architecture(name, num_classes):
    if name == "PretrainedViTB16":
        return get_pretrained_vit_b16(num_classes)
    elif name == "PretrainedViTB32":
        return get_pretrained_vit_b32(num_classes)
    elif name == "PretrainedViTL16":
        return get_pretrained_vit_l16(num_classes)
    elif name == "PretrainedViTL32":
        return get_pretrained_vit_l32(num_classes)
    elif name == "PretrainedViTB16Imagenet1k":
        return get_pretrained_vit_b16_imagenet1k(num_classes)
    elif name == "PretrainedViTB32Imagenet1k":
        return get_pretrained_vit_b32_imagenet1k(num_classes)
    elif name == "PretrainedViTL16Imagenet1k":
        return get_pretrained_vit_l16_imagenet1k(num_classes)
    elif name == "PretrainedViTL32Imagenet1k":
        return get_pretrained_vit_b32_imagenet1k(num_classes)


def get_pretrained_vit_b16(num_classes=2):
    return get_pretrained_vit("B_16", num_classes)


def get_pretrained_vit_b32(num_classes=2):
    return get_pretrained_vit("B_32", num_classes)


def get_pretrained_vit_l16(num_classes=2):
    return get_pretrained_vit("L_16", num_classes)


def get_pretrained_vit_l32(num_classes=2):
    return get_pretrained_vit("L_32", num_classes)


def get_pretrained_vit_l16_imagenet1k(num_classes=2):
    return get_pretrained_vit("L_16_imagenet1k", num_classes)


def get_pretrained_vit_l32_imagenet1k(num_classes=2):
    return get_pretrained_vit("L_32_imagenet1k", num_classes)


def get_pretrained_vit_b16_imagenet1k(num_classes=2):
    return get_pretrained_vit("B_16_imagenet1k", num_classes)


def get_pretrained_vit_b32_imagenet1k(num_classes=2):
    return get_pretrained_vit("B_32_imagenet1k", num_classes)


def get_pretrained_vit(model_name, num_classes):
    vit = ViT_pretrained(name=model_name, pretrained=True)
    in_features = vit.fc.in_features
    vit.fc = nn.Linear(in_features=in_features, out_features=num_classes)
    vit.init_weights()
    return vit

