import torch
import torch.nn as nn
import torchvision.models as models

# This file contains a list of wrappers to PyTorch pretrained models
# Use these wrappers for obtaining the version of networks with a different number of outs
# list of pretrained models: https://pytorch.org/docs/stable/torchvision/models.html


def check_name(name):
    arch_list = ['mobilenetv2', 'densenet121', 'densenet161', 'squeezenet', 'resnet34', 'resnet50', 'resnet101',
                 'inceptionv3', 'vgg16', 'resnext', 'efficientnetv2-s', 'efficientnetv2-m', 'efficientnetv2-l',
                 'convnext_small', 'convnext_base',  'convnext_large',
                 'swin_v2_s', 'swin_v2_b', 'swin_v2_t']
    assert name in arch_list, f"Architecture name = {name}. Not in {arch_list}!"


def get_vram_gb():
    return round(torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024)


def get_target_layer(name, net):
    check_name(name)
    if name == 'mobilenetv2':
        return [net.features[-1]]
    elif name == 'densenet161':
        return [net[-5]]   # return [net.features[-1]]
    elif name == 'densenet121':
        return [net[-5]]  # return [net.features[-1]]
    elif name == 'squeezenet':
        return [net.features[-1]]
    elif name == 'resnet34':
        return [net.layer4[-1]]
    elif name == 'resnet50':
        return [net.layer4[-1]]
    elif name == 'resnet101':
        return [net.layer4[-1]]
    elif name == 'inceptionv3':
        return [net.Mixed_7c.branch_pool.bn]
    elif name == 'vgg16':
        return [net.features[-1]]
    elif name == 'resnext':
        return [net.layer4[-1]]
    elif name == 'efficientnetv2-s':
        return [net.features[-1]]
    elif name == 'efficientnetv2-m':
        return [net.features[-1]]
    elif name == 'efficientnetv2-l':
        return [net.features[-1]]
    elif name == 'SmallViT':
        return [net.transformer]
    elif name == "PretrainedViT":
        return [net.transformer]
    elif name == 'convnext_small':
        return [net.features[-1][-1]]
    elif name == 'convnext_base':
        return [net.features[-1][-1]]
    elif name == 'convnext_large':
        return [net.features[-1][-1]]
    elif name == 'swin_v2_s':
        return [net.features[-1].blocks[-1]]
    elif name == 'swin_v2_b':
        return [net.features[-1].blocks[-1]]
    elif name == 'swin_v2_t':
        return [net.features[-1].blocks[-1]]


def get_architecture(name, num_classes, pretrained, feature_extract=False):
    check_name(name)
    if name == 'mobilenetv2':
        return get_mobilenet_v2(num_classes, pretrained, feature_extract=feature_extract)
    elif name == 'densenet161':
        return get_densenet161(num_classes, pretrained, feature_extract=feature_extract)
    elif name == 'densenet121':
        return get_densenet121(num_classes, pretrained, feature_extract=feature_extract)
    elif name == 'squeezenet':
        return get_squeezenet(num_classes, pretrained, feature_extract=feature_extract)
    elif name == 'resnet34':
        return get_resnet34(num_classes, pretrained, feature_extract=feature_extract)
    elif name == 'resnet50':
        return get_resnet50(num_classes, pretrained, feature_extract=feature_extract)
    elif name == 'resnet101':
        return get_resnet101(num_classes, pretrained, feature_extract=feature_extract)
    elif name == 'inceptionv3':
        return get_inception_v3(num_classes, pretrained, feature_extract=feature_extract)
    elif name == 'vgg16':
        return get_vgg16(num_classes, pretrained, feature_extract=feature_extract)
    elif name == 'resnext':
        return get_resnext(num_classes, pretrained, feature_extract=feature_extract)
    elif name == 'efficientnetv2-s':
        return get_efficientnet_v2_s(num_classes, pretrained, feature_extract=feature_extract)
    elif name == 'efficientnetv2-m':
        return get_efficientnet_v2_m(num_classes, pretrained, feature_extract=feature_extract)
    elif name == 'efficientnetv2-l':
        return get_efficientnet_v2_l(num_classes, pretrained, feature_extract=feature_extract)
    elif name == 'convnext_small':
        return get_convnext_small(num_classes, pretrained, feature_extract=feature_extract)
    elif name == 'convnext_base':
        return get_convnext_base(num_classes, pretrained, feature_extract=feature_extract)
    elif name == 'convnext_large':
        return get_convnext_large(num_classes, pretrained, feature_extract=feature_extract)
    elif name == 'swin_v2_s':
        return get_swin_v2_s(num_classes, pretrained, feature_extract=feature_extract)
    elif name == 'swin_v2_b':
        return get_swin_v2_b(num_classes, pretrained, feature_extract=feature_extract)
    elif name == 'swin_v2_t':
        return get_swin_v2_t(num_classes, pretrained, feature_extract=feature_extract)


def get_pretrained_architecture_feature_extractor(name):
    check_name(name)
    if name == 'mobilenetv2':
        return get_mobilenet_v2_feature_extractor()
    elif name == 'densenet161':
        return get_densenet161_feature_extractor()
    elif name == 'densenet121':
        return get_densenet121_feature_extractor()
    elif name == 'squeezenet':
        return get_squeezenet_feature_extractor()
    elif name == 'resnet34':
        return get_resnet34_feature_extractor()
    elif name == 'resnet50':
        return get_resnet50_feature_extractor()
    elif name == 'resnet101':
        return get_resnet101_feature_extractor()
    elif name == 'inceptionv3':
        return get_inception_v3_feature_extractor()
    elif name == 'vgg16':
        return get_vgg16_feature_extractor()
    elif name == 'resnext':
        return get_resnext_feature_extractor()
    elif name == 'efficientnetv2-s':
        return get_efficientnet_v2_s_feature_extractor()
    elif name == 'efficientnetv2-m':
        return get_efficientnet_v2_m_feature_extractor()
    elif name == 'efficientnetv2-l':
        return get_efficientnet_v2_l_feature_extractor()
    elif name == 'convnext_small':
        return get_convnext_small_feature_extractor()
    elif name == 'convnext_base':
        return get_convnext_base_feature_extractor()
    elif name == 'convnext_large':
        return get_convnext_large_feature_extractor()
    elif name == 'swin_v2_s':
        return get_swin_v2_s_feature_extractor()
    elif name == 'swin_v2_b':
        return get_swin_v2_b_feature_extractor()
    elif name == 'swin_v2_t':
        return get_swin_v2_t_feature_extractor()


def get_trained_architecture_feature_extractor(name, device, net_path, num_classes):
    check_name(name)
    if name == 'mobilenetv2':
        return get_trained_mobilenet_v2_feature_extractor(device, net_path, num_classes)
    elif name == 'densenet161':
        return get_trained_densenet161_feature_extractor(device, net_path, num_classes)
    elif name == 'densenet121':
        return get_trained_densenet121_feature_extractor(device, net_path, num_classes)
    elif name == 'squeezenet':
        return get_trained_squeezenet_feature_extractor(device, net_path, num_classes)
    elif name == 'resnet34':
        return get_trained_resnet34_feature_extractor(device, net_path, num_classes)
    elif name == 'resnet50':
        return get_trained_resnet50_feature_extractor(device, net_path, num_classes)
    elif name == 'resnet101':
        return get_trained_resnet101_feature_extractor(device, net_path, num_classes)
    elif name == 'inceptionv3':
        return get_trained_inception_v3_feature_extractor(device, net_path, num_classes)
    elif name == 'vgg16':
        return get_trained_vgg16_feature_extractor(device, net_path, num_classes)
    elif name == 'resnext':
        return get_trained_resnext_feature_extractor(device, net_path, num_classes)
    elif name == 'efficientnetv2-s':
        return get_trained_efficientnet_v2_s_feature_extractor(device, net_path, num_classes)
    elif name == 'efficientnetv2-m':
        return get_trained_efficientnet_v2_m_feature_extractor(device, net_path, num_classes)
    elif name == 'efficientnetv2-l':
        return get_trained_efficientnet_v2_l_feature_extractor(device, net_path, num_classes)
    elif name == 'convnext_small':
        return get_trained_convnext_small_feature_extractor(device, net_path, num_classes)
    elif name == 'convnext_base':
        return get_trained_convnext_base_feature_extractor(device, net_path, num_classes)
    elif name == 'convnext_large':
        return get_trained_convnext_large_feature_extractor(device, net_path, num_classes)
    elif name == 'swin_v2_s':
        return get_trained_swin_v2_s_feature_extractor(device, net_path, num_classes)
    elif name == 'swin_v2_b':
        return get_trained_swin_v2_b_feature_extractor(device, net_path, num_classes)
    elif name == 'swin_v2_t':
        return get_trained_swin_v2_t_feature_extractor(device, net_path, num_classes)


def get_efficientnet_v2_s(num_classes, pretrained=True, feature_extract=False):
    weights = models.EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
    efficientnet = models.efficientnet_v2_s(weights=weights)
    efficientnet.name = 'efficientnetv2-s'
    efficientnet.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=False),
        nn.Linear(in_features=1280, out_features=num_classes),
    )
    return efficientnet


def get_convnext_small(num_classes, pretrained=True, feature_extract=False):
    weights = models.ConvNeXt_Small_Weights.DEFAULT if pretrained else None
    model = models.convnext_small(weights=weights)
    model.name = 'convnext_small'
    if feature_extract:
        for p in model.features.parameters():
            p.requires_grad = False
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, num_classes)
    return model


def get_convnext_base(num_classes, pretrained, feature_extract=False):
    weights = models.ConvNeXt_Base_Weights.DEFAULT if pretrained else None
    model = models.convnext_base(weights=weights)
    model.name = 'convnext_base'
    if feature_extract:
        for p in model.features.parameters():
            p.requires_grad = False
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, num_classes)
    return model


def get_convnext_large(num_classes, pretrained, feature_extract=False):
    weights = models.ConvNeXt_Large_Weights.DEFAULT if pretrained else None
    model = models.convnext_large(weights=weights)
    model.name = 'convnext_large'
    if feature_extract:
        for p in model.features.parameters():
            p.requires_grad = False
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, num_classes)
    return model


def get_swin_v2_t(num_classes, pretrained, feature_extract=False):
    weights = models.Swin_V2_T_Weights.DEFAULT if pretrained else None
    model = models.swin_v2_t(weights=weights)
    model.name = 'swin_v2_t'
    if feature_extract:
        for p in model.parameters():
            p.requires_grad = False
    in_features = model.head.in_features
    model.head = nn.Linear(in_features, num_classes)
    return model


def get_swin_v2_s(num_classes, pretrained, feature_extract=False):
    weights = models.Swin_V2_S_Weights.DEFAULT if pretrained else None
    model = models.swin_v2_s(weights=weights)
    model.name = 'swin_v2_s'
    if feature_extract:
        for p in model.parameters():
            p.requires_grad = False
    in_features = model.head.in_features
    model.head = nn.Linear(in_features, num_classes)
    return model


def get_swin_v2_b(num_classes, pretrained, feature_extract=False):
    weights = models.Swin_V2_B_Weights.DEFAULT if pretrained else None
    model = models.swin_v2_b(weights=weights)
    model.name = 'swin_v2_b'
    if feature_extract:
        for p in model.parameters():
            p.requires_grad = False
    in_features = model.head.in_features
    model.head = nn.Linear(in_features, num_classes)
    return model


def get_efficientnet_v2_m(num_classes, pretrained=True, feature_extract=False):
    weights = models.EfficientNet_V2_M_Weights.DEFAULT if pretrained else None
    efficientnet = models.efficientnet_v2_m(weights=weights)
    efficientnet.name = 'efficientnetv2-m'
    efficientnet.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=False),
        nn.Linear(in_features=1280, out_features=num_classes),
    )
    return efficientnet


def get_efficientnet_v2_l(num_classes, pretrained=True, feature_extract=False):
    weights = models.EfficientNet_V2_L_Weights.DEFAULT if pretrained else None
    efficientnet = models.efficientnet_v2_l(weights=weights)
    efficientnet.name = 'efficientnetv2-l'
    efficientnet.classifier = nn.Sequential(
        nn.Dropout(p=0.4, inplace=False),
        nn.Linear(in_features=1280, out_features=num_classes),
    )
    return efficientnet


def get_mobilenet_v2(num_classes, pretrained=True, feature_extract=False):
    weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
    mobilenet = models.mobilenet_v2(weights=weights)
    mobilenet.name = 'mobilenetv2'
    # Change last layer with out_features = num_classes
    mobilenet.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=False),
        nn.Linear(in_features=1280, out_features=num_classes),
    )
    return mobilenet


def densenet_inplace_handle(densenet, inplace=False):
    if not inplace:
        densenet = nn.Sequential(
            densenet.features,
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(1),
            densenet.classifier
        )
    return densenet


def get_densenet161(num_classes, pretrained=True, feature_extract=False, inplace=False):
    weights = models.DenseNet161_Weights.DEFAULT if pretrained else None
    densenet = models.densenet161(weights=weights)
    densenet.name = 'densenet161'
    # Change last layer with out_features = num_classes
    densenet.classifier = nn.Linear(in_features=2208, out_features=num_classes, bias=True)
    densenet = densenet_inplace_handle(densenet, inplace=inplace)
    return densenet


def get_vgg16(num_classes, pretrained, feature_extract=False):
    weights = models.VGG16_Weights.DEFAULT if pretrained else None
    vgg = models.vgg16(weights=weights)
    vgg.name = 'vgg16'
    vgg.classifier = nn.Sequential(
        nn.Linear(in_features=25088, out_features=4096, bias=True),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5, inplace=False),
        nn.Linear(in_features=4096, out_features=4096, bias=True),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5, inplace=False),
        nn.Linear(in_features=4096, out_features=num_classes, bias=True)
    )
    return vgg


def get_resnext(num_classes, pretrained, feature_extract=False):
    weights = models.ResNeXt50_32X4D_Weights.DEFAULT if pretrained else None
    resnext = models.resnext50_32x4d(weights=weights)
    resnext.name = 'resnext'
    resnext.fc = nn.Linear(in_features=resnext.fc.in_features, out_features=num_classes, bias=True)
    return resnext


def get_densenet121(num_classes, pretrained=True, feature_extract=False, inplace=False):
    weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
    densenet = models.densenet121(weights=weights)
    densenet.name = 'densenet121'
    # Change last layer with out_features = num_classes
    densenet.classifier = nn.Linear(in_features=1024, out_features=num_classes, bias=True)
    densenet = densenet_inplace_handle(densenet, inplace=inplace)
    return densenet


def get_squeezenet(num_classes, pretrained=True, feature_extract=False):
    weights = models.SqueezeNet1_0_Weights.DEFAULT if pretrained else None
    squeezenet = models.squeezenet1_0(weights=weights)
    squeezenet.name = 'squeezenet'
    squeezenet.classifier = nn.Sequential(
        nn.Dropout(p=0.5, inplace=False),
        nn.Conv2d(512, num_classes, kernel_size=(1,1),stride=(1,1)),
        nn.ReLU(inplace=True),
        nn.AdaptiveAvgPool2d(output_size=(1,1)),
    )
    return squeezenet


def get_resnet34(num_classes, pretrained=True, feature_extract=False):
    weights = models.ResNet34_Weights.DEFAULT if pretrained else None
    resnet = models.resnet34(weights=weights)
    resnet.name = 'resnet34'
    resnet.fc = nn.Linear(in_features=resnet.fc.in_features, out_features=num_classes, bias=True)
    return resnet


def get_resnet50(num_classes, pretrained=True, feature_extract=False):
    weights = models.ResNet50_Weights.DEFAULT if pretrained else None
    resnet = models.resnet50(weights=weights)
    resnet.name = 'resnet50'
    resnet.fc = nn.Linear(in_features=resnet.fc.in_features, out_features=num_classes, bias=True)
    return resnet

def get_resnet101(num_classes, pretrained=True, feature_extract=False):
    weights = models.ResNet101_Weights.DEFAULT if pretrained else None
    resnet = models.resnet101(weights=weights)
    resnet.name = 'resnet101'
    resnet.fc = nn.Linear(in_features=resnet.fc.in_features, out_features=num_classes, bias=True)
    return resnet


def get_inception_v3(num_classes, pretrained=True, feature_extract=False):
    weights = models.Inception_V3_Weights.DEFAULT if pretrained else None
    inception = models.inception_v3(weights=weights)
    inception.name = 'inceptionv3'
    set_parameter_requires_grad(inception, feature_extract)
    # Handle the auxiliary net
    num_features = inception.AuxLogits.fc.in_features
    inception.AuxLogits.fc = nn.Linear(in_features=num_features, out_features=num_classes, bias=True)
    # Handle the primary net
    num_features = inception.fc.in_features
    inception.fc = nn.Linear(in_features=num_features, out_features=num_classes, bias=True)
    return inception


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def get_resnet_feature_extractor(resnet):
    modules = list(resnet.children())[:-1]
    resnet = nn.Sequential(*modules)
    for p in resnet.parameters():
        p.requires_grad = False
    return resnet


def get_resnet34_feature_extractor():
    resnet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
    return get_resnet_feature_extractor(resnet)


def get_resnet50_feature_extractor():
    resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    return get_resnet_feature_extractor(resnet)


def get_resnet101_feature_extractor():
    resnet = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
    return get_resnet_feature_extractor(resnet)


def get_vgg16_feature_extractor():
    net = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
    # net.classifier = nn.Identity()
    net.classifier = net.classifier[:2]
    for p in net.parameters():
        p.requires_grad = False
    return net


def get_resnext_feature_extractor():
    net = models.resnext50_32x4d(weights=models.ResNeXt50_32X4D_Weights.DEFAULT)
    fc_to_feature_extractor(net)
    return net


def get_inception_v3_feature_extractor():
    net = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
    net.fc = nn.Identity()
    for p in net.parameters():
        p.requires_grad = False
    return net


def squeezenet_to_feature_extractor(net):
    net.classifier[1] = nn.Identity()
    for p in net.parameters():
        p.requires_grad = False


def classifier_to_feature_extractor(net):
    net.classifier = nn.Identity()
    for p in net.parameters():
        p.requires_grad = False


def fc_to_feature_extractor(net):
    net.fc = nn.Identity()
    for p in net.parameters():
        p.requires_grad = False


def vgg_to_feature_extractor(net):
    net.classifier = net.classifier[:2]
    for p in net.parameters():
        p.requires_grad = False


def get_squeezenet_feature_extractor():
    squeezenet = models.squeezenet1_0(weights=models.SqueezeNet1_0_Weights.DEFAULT)
    squeezenet_to_feature_extractor(squeezenet)
    return squeezenet


def get_mobilenet_v2_feature_extractor():
    mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    classifier_to_feature_extractor(mobilenet)
    return mobilenet


def get_efficientnet_v2_s_feature_extractor():
    efficientnet = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT)
    classifier_to_feature_extractor(efficientnet)
    return efficientnet


def get_efficientnet_v2_m_feature_extractor():
    efficientnet = models.efficientnet_v2_m(weights=models.EfficientNet_V2_M_Weights.DEFAULT)
    classifier_to_feature_extractor(efficientnet)
    return efficientnet


def get_efficientnet_v2_l_feature_extractor():
    efficientnet = models.efficientnet_v2_l(weights=models.EfficientNet_V2_L_Weights.DEFAULT)
    classifier_to_feature_extractor(efficientnet)
    return efficientnet


def get_convnext_small_feature_extractor():
    convnext = models.convnext_small(weights=models.ConvNeXt_Small_Weights.DEFAULT)
    classifier_to_feature_extractor(convnext)
    return convnext


def get_convnext_base_feature_extractor():
    convnext = models.convnext_base(weights=models.ConvNeXt_Base_Weights.DEFAULT)
    classifier_to_feature_extractor(convnext)
    return convnext


def get_convnext_large_feature_extractor():
    convnext = models.convnext_large(weights=models.ConvNeXt_Large_Weights.DEFAULT)
    classifier_to_feature_extractor(convnext)
    return convnext


def get_swin_v2_s_feature_extractor():
    model = models.swin_v2_s(weights=models.Swin_V2_S_Weights.DEFAULT)
    classifier_to_feature_extractor(model)
    return model


def get_swin_v2_t_feature_extractor():
    model = models.swin_v2_t(weights=models.Swin_V2_T_Weights.DEFAULT)
    classifier_to_feature_extractor(model)
    return model


def get_swin_v2_b_feature_extractor():
    model = models.swin_v2_b(weights=models.Swin_V2_B_Weights.DEFAULT)
    classifier_to_feature_extractor(model)
    return model


def get_densenet121_feature_extractor():
    densenet = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
    classifier_to_feature_extractor(densenet)
    return densenet


def get_densenet161_feature_extractor():
    densenet = models.densenet161(weights=models.DenseNet161_Weights.DEFAULT)
    classifier_to_feature_extractor(densenet)
    return densenet


def get_trained_inception_v3_feature_extractor(device, net_path, num_classes):
    net = get_inception_v3(num_classes, pretrained=False)
    net = load_model(net, device, net_path)
    fc_to_feature_extractor(net)
    return net


def get_trained_vgg16_feature_extractor(device, net_path, num_classes):
    net = get_vgg16(num_classes, pretrained=False)
    net = load_model(net, device, net_path)
    vgg_to_feature_extractor(net)
    return net


def get_trained_resnext_feature_extractor(device, net_path, num_classes):
    net = get_resnext(num_classes, pretrained=False)
    net = load_model(net, device, net_path)
    fc_to_feature_extractor(net)
    return net


def get_trained_resnet50_feature_extractor(device, net_path, num_classes):
    resnet = get_resnet50(num_classes, pretrained=False)
    resnet = load_model(resnet, device, net_path)
    return get_resnet_feature_extractor(resnet)


def get_trained_resnet101_feature_extractor(device, net_path, num_classes):
    resnet = get_resnet101(num_classes, pretrained=False)
    resnet = load_model(resnet, device, net_path)
    return get_resnet_feature_extractor(resnet)


def get_trained_resnet34_feature_extractor(device, net_path, num_classes):
    resnet = get_resnet34(num_classes, pretrained=False)
    resnet = load_model(resnet, device, net_path)
    return get_resnet_feature_extractor(resnet)


def get_trained_squeezenet_feature_extractor(device, net_path, num_classes):
    squeezenet = get_squeezenet(num_classes, pretrained=False)
    squeezenet = load_model(squeezenet, device, net_path)
    squeezenet_to_feature_extractor(squeezenet)
    return squeezenet


def get_trained_densenet121_feature_extractor(device, net_path, num_classes):
    densenet = get_densenet121(num_classes, pretrained=False)
    densenet = load_model(densenet, device, net_path)
    classifier_to_feature_extractor(densenet)
    return densenet


def get_trained_densenet161_feature_extractor(device, net_path, num_classes):
    densenet = get_densenet161(num_classes, pretrained=False)
    densenet = load_model(densenet, device, net_path)
    classifier_to_feature_extractor(densenet)
    return densenet


def get_trained_mobilenet_v2_feature_extractor(device, net_path, num_classes):
    mobilenet = get_mobilenet_v2(num_classes, pretrained=False)
    mobilenet = load_model(mobilenet, device, net_path)
    classifier_to_feature_extractor(mobilenet)
    return mobilenet


def get_trained_efficientnet_v2_s_feature_extractor(device, net_path, num_classes):
    efficientnet = get_efficientnet_v2_s(num_classes, pretrained=False)
    efficientnet = load_model(efficientnet, device, net_path)
    classifier_to_feature_extractor(efficientnet)
    return efficientnet


def get_trained_efficientnet_v2_m_feature_extractor(device, net_path, num_classes):
    efficientnet = get_efficientnet_v2_m(num_classes, pretrained=False)
    efficientnet = load_model(efficientnet, device, net_path)
    classifier_to_feature_extractor(efficientnet)
    return efficientnet


def get_trained_efficientnet_v2_l_feature_extractor(device, net_path, num_classes):
    efficientnet = get_efficientnet_v2_l(num_classes, pretrained=False)
    efficientnet = load_model(efficientnet, device, net_path)
    classifier_to_feature_extractor(efficientnet)
    return efficientnet


def get_trained_convnext_small_feature_extractor(device, net_path, num_classes):
    convnext = get_convnext_small(num_classes, pretrained=False)
    convnext = load_model(convnext, device, net_path)
    classifier_to_feature_extractor(convnext)
    return convnext


def get_trained_convnext_base_feature_extractor(device, net_path, num_classes):
    convnext = get_convnext_base(num_classes, pretrained=False)
    convnext = load_model(convnext, device, net_path)
    classifier_to_feature_extractor(convnext)
    return convnext


def get_trained_convnext_large_feature_extractor(device, net_path, num_classes):
    convnext = get_convnext_large(num_classes, pretrained=False)
    convnext = load_model(convnext, device, net_path)
    classifier_to_feature_extractor(convnext)
    return convnext


def get_trained_swin_v2_s_feature_extractor(device, net_path, num_classes):
    swin = get_swin_v2_s(num_classes, pretrained=False)
    swin = load_model(swin, device, net_path)
    classifier_to_feature_extractor(swin)
    return swin


def get_trained_swin_v2_b_feature_extractor(device, net_path, num_classes):
    swin = get_swin_v2_b(num_classes, pretrained=False)
    swin = load_model(swin, device, net_path)
    classifier_to_feature_extractor(swin)
    return swin


def get_trained_swin_v2_t_feature_extractor(device, net_path, num_classes):
    swin = get_swin_v2_t(num_classes, pretrained=False)
    swin = load_model(swin, device, net_path)
    classifier_to_feature_extractor(swin)
    return swin


def load_model(net, device, net_path):
    net = torch.load(net_path)
    net = net.to(device)
    return net


def load_state_dict(net, device, net_path):
    net.load_state_dict(torch.load(net_path))
    net = net.to(device)
    net.eval()
    return net


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_layers(model):
    l = [module for module in model.modules() if not isinstance(module, nn.Sequential)]
    return len(l)
