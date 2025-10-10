import torch
from pytorch_grad_cam import (GradCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, EigenGradCAM, LayerCAM)
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision.transforms import Compose, ToTensor

from classification.gutils.net_adapter import get_target_layer
from classification.gutils.utils import concat_tile, vconcat, hconcat
import cv2
import os
import numpy as np

from classification.xai.utils import reshape_transform


def preprocess_image_base(img: np.ndarray) -> torch.Tensor:
    preprocessing = Compose([
        ToTensor(),
    ])
    return preprocessing(img.copy()).unsqueeze(0)


def preprocess_and_inference(rgb_img, net, dsize, device):
    rgb_img = cv2.resize(rgb_img, dsize)
    input_tensor = preprocess_image_base(rgb_img)
    rgb_img = np.float32(rgb_img) / 255
    input_tensor = input_tensor.to(device)
    print(f"Input tensor shape = {input_tensor.shape}")
    net.to(device)
    net.eval()
    with torch.no_grad():
        output_tensor = net(input_tensor)
        output_class = torch.argmax(output_tensor, dim=1)
        output_class_item = output_class.cpu().numpy().item()
    return rgb_img, input_tensor, output_class_item


def create_cam_visualization(cam, rgb_img, input_tensor, target_category, aug_smooth, eig_smooth, original_size):
    width, height = original_size
    grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category,
                        aug_smooth=aug_smooth, eigen_smooth=eig_smooth)
    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    visualization = cv2.resize(visualization, (width, height))
    return visualization


def grad_cam_base(rgb_image, net, net_name, label_to_study, device, dsize=(227, 227)):
    aug_smooth = True
    eig_smooth = True

    target_layer = get_target_layer(net_name, net)
    height, width, _ = rgb_image.shape
    rgb_img, input_tensor, output_class_item = preprocess_and_inference(rgb_image, net, dsize, device)
    cam = GradCAM(model=net, target_layers=target_layer, use_cuda=True)
    visualization = create_cam_visualization(cam, rgb_img, input_tensor, label_to_study, aug_smooth, eig_smooth, (width, height))
    vis_bgr = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)
    return vis_bgr


def grad_cam_inference(net, net_name, label_targets, ext_ids, grad_cam_dir, classes_encoding, device,
                       image_dir=None, dsize=(227, 227), put_text=False):
    # ---Color of the border---
    black = [0, 0, 0]
    white = [255, 255, 255]
    pad = 5

    net_type = str(type(net))
    target_layer = get_target_layer(net_name, net)

    methods = {
        "gradcam": GradCAM,
        "gradcam++": GradCAMPlusPlus,
        "ablationcam": AblationCAM,
        "xgradcam": XGradCAM,
        "eigencam": EigenCAM,
        "eigengradcam": EigenGradCAM,
        "layercam": LayerCAM
    }

    aug_smooth = True
    eig_smooth = True
    # grad_cam_dir = os.path.join(grad_cam_dir, f'label_{label_targets}')
    # print(f"grad_cam_dir: {grad_cam_dir}")
    # os.makedirs(grad_cam_dir, exist_ok=True)

    res_trans = reshape_transform if "ViT" in net_name else None

    for path_img in ext_ids:
        if image_dir:
            path_img = os.path.join(image_dir, path_img)
        print(f"Opening image {path_img}")
        if not os.path.exists(path_img):
            print("Image not exists!")
            continue

        rgb_img = cv2.cvtColor(cv2.imread(path_img, 1), cv2.COLOR_BGR2RGB)

        height, width, _ = rgb_img.shape
        im1 = rgb_img.copy()
        rgb_img, input_tensor, output_class_item = preprocess_and_inference(rgb_img, net, dsize, device)
        output_class_name = classes_encoding[output_class_item]
        print(f"Output Class: {output_class_item}, Label: {output_class_name}")

        img_grd = []
        for k, method in methods.items():
            # , use_cuda = True
            cam = method(model=net, target_layers=target_layer, reshape_transform=res_trans)
            if isinstance(label_targets, list):
                targets = [ClassifierOutputTarget(i) for i in label_targets]
            else:
                targets = [ClassifierOutputTarget(label_targets)]
            print(f"Label targets: {label_targets}")
            try:
                kwargs = {
                    "input_tensor": input_tensor,
                    "targets": targets,
                    "aug_smooth": aug_smooth,
                    "eigen_smooth": eig_smooth,
                }
                grayscale_cam = cam(**kwargs)
                grayscale_cam = grayscale_cam[0, :]
                visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
                visualization = cv2.resize(visualization, (width, height))
                if put_text:
                    visualization = cv2.putText(visualization, k, org=(10, 70), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                                fontScale=1, color=(255, 255, 255))

                vis_border = cv2.copyMakeBorder(visualization, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=white)
                img_grd.append(vis_border)
            except np.linalg.LinAlgError as E:
                print(f"np.linalg.LinAlgError: {E}")

        im_border = cv2.copyMakeBorder(im1, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=white)
        im_v = hconcat([im_border, img_grd[0], img_grd[1], img_grd[2], img_grd[3], img_grd[4], img_grd[5], img_grd[6]])

        im_v = cv2.cvtColor(im_v, cv2.COLOR_RGB2BGR)

        img_name = os.path.basename(path_img)

        if aug_smooth and not eig_smooth:
            cv2.imwrite(os.path.join(grad_cam_dir, f'A_smooth_gcam_{net_name}_{img_name}.png'), im_v)
        elif not aug_smooth and eig_smooth:
            cv2.imwrite(os.path.join(grad_cam_dir, f'E_smooth_gcam_{net_name}_{img_name}.png'), im_v)
        elif aug_smooth and eig_smooth:
            cv2.imwrite(os.path.join(grad_cam_dir, f'AE_smooth_gcam_{net_name}_{img_name}.png'), im_v)
        else:
            cv2.imwrite(os.path.join(grad_cam_dir, f'gcam_{net_name}_{img_name}.png'), im_v)
