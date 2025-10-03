import os
import random
import segmentation_models_pytorch as smp
import torch
import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from tabulate import tabulate
from tqdm import tqdm


def imoverlay(img, mask, alpha=0.5):
    return cv2.addWeighted(mask, alpha, img, 1-alpha, 0)


def run_segmentation(pl_model, dataloader, binary_dir, multiclass_dir, class_names, device='cuda'):
    # Iterate over tiles, run prediction, save output masks
    for batch in dataloader:
        image = batch['image'].to(device)
        with torch.no_grad():
            output = pl_model(image)

        for ii, el in enumerate(image):
            pred_mask = torch.argmax(output, 1)[ii]  #.unsqueeze(1)
            pred_img_array = pred_mask.cpu().numpy().astype('uint8') #.squeeze()

            # Save multiclass masks in 'Multiclass'
            tile_name = str(batch['filename'][ii])
            output_filename_multiclass = tile_name + ".png"
            output_path_multiclass = os.path.join(multiclass_dir, output_filename_multiclass)
            print(f'Saving multiclass mask to: {output_path_multiclass}')
            pred_img = Image.fromarray(pred_img_array)
            pred_img.save(output_path_multiclass)

            # Create a subdir for current tile in 'Binary'
            tile_subdir = os.path.join(binary_dir, tile_name)
            if not os.path.exists(tile_subdir):
                os.makedirs(tile_subdir)

            # Save binary masks in subdir
            for i, class_name in enumerate(class_names):
                binary_mask = (pred_img_array == i).astype('uint8') * 255
                binary_img = Image.fromarray(binary_mask)
                output_filename_binary = f"{class_name}.png"
                output_path_binary = os.path.join(tile_subdir, output_filename_binary)
                print(f'Saving {class_name} mask to: {output_path_binary}')
                binary_img.save(output_path_binary)


def run_segmentation_binary(model, dataloader, output_dir, device='cuda'):
    # Inference
    for b, batch in enumerate(dataloader):
        print(f"Batch: {b+1:3d}/{len(dataloader):3d}")
        image = batch['image'].to(device)
        with torch.no_grad():
            pr_masks = model.predict(image)

        for i, el in enumerate(image):
            original_filename = batch["filename"][i]
            mask_filename = original_filename + ".png"
            mask_output_path = os.path.join(output_dir, mask_filename)
            Image.fromarray((pr_masks[i].squeeze().cpu().numpy() * 255).astype(np.uint8)).save(mask_output_path)
            print(f"Saved mask to {mask_output_path}")


def run_segmentation_multiclass(pl_model, subset_dl, device, classes, calc_loss=False):
    with torch.no_grad():
        outputs = []
        test_loss = 0.0

        for batch in tqdm(subset_dl):
            image = batch['image'].to(device)
            mask = batch['mask'].to(device)
            filename = batch['filename']
            output = pl_model(image)
            pred = torch.argmax(output, 1).unsqueeze(1)
            tp, fp, fn, tn = smp.metrics.get_stats(pred, mask.long(), mode='multiclass', num_classes=classes)
            outputs.append({"filename": filename, "tp": tp, "fp": fp, "fn": fn, "tn": tn})
            if calc_loss:
                loss = pl_model.loss_fn(output, mask.long())
                test_loss += loss.item()

        filename = [x["filename"] for x in outputs]
        tp = torch.cat([x["tp"] for x in outputs]).long()
        fp = torch.cat([x["fp"] for x in outputs]).long()
        fn = torch.cat([x["fn"] for x in outputs]).long()
        tn = torch.cat([x["tn"] for x in outputs]).long()

        print(f'Test Loss, calc={calc_loss}: {test_loss / len(subset_dl)}')
        print('IoU:', smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise").item())

        torch_obj = {"filename": filename, "tp": tp, "fp": fp, "fn": fn, "tn": tn}
    return torch_obj


def compute_metrics_all(tp, fp, fn, tn):
    print(f"TP, FP, FN, TN shape: {tp.shape}, {fp.shape}, {fn.shape}, {tn.shape}")
    tpa = torch.sum(tp, 0)
    fpa = torch.sum(fp, 0)
    fna = torch.sum(fn, 0)
    tna = torch.sum(tn, 0)
    print(f"TPa, FPa, FNa, TNa shape: {tpa.shape}, {fpa.shape}, {fna.shape}, {tna.shape}")
    recall_a = tpa / (tpa + fna)
    precision_a = tpa / (tpa + fpa)
    dice_a = 2 * tpa / (2 * tpa + fpa + fna)
    iou_a = tpa / (tpa + fpa + fna)
    metrics_a = torch.stack([recall_a, precision_a, dice_a, iou_a]).numpy()
    return metrics_a


def save_table_metrics(logs_dir, subset, metrics_list, names_list, class_names):
    for metrics_rounded, metrics_name in zip(metrics_list, names_list):
        info = {'Metrics': ['Recall', 'Precision', 'Dice', 'IoU']}
        info.update(
            {class_names[i]: metrics_rounded[:, i] for i in range(len(class_names))}
        )

        table = tabulate(info, headers='keys', tablefmt='simple')
        # table = tabulate(info, headers='keys', tablefmt='fancy_grid')
        print(table)

        table_out = os.path.join(logs_dir, f'table_{subset}_{metrics_name}.txt')
        with open(table_out, 'w', encoding='utf-8') as f:
            f.write(table)


def export_sample_output_images(pl_model, subset_dl, arch, enc_name, device, fig_output_dir, palette):
    pal = [value for color in palette for value in color]

    for batch in subset_dl:

        mask_batch = batch['mask']
        image_batch = batch['image']
        filename_batch = batch['filename']

        with torch.no_grad():
            image_batch = image_batch.to(device)
            pmap = pl_model(image_batch)
            pred_batch = torch.argmax(pmap, 1)

        for image_t, mask_t, pred_t, filename in zip(image_batch, mask_batch, pred_batch, filename_batch):
            cols = ['Image', 'Mask', 'Prediction']
            fig, axes = plt.subplots(1, 3, sharex='row', sharey='row',
                                     subplot_kw={'xticks': [], 'yticks': []},
                                     tight_layout=True, figsize=(30, 12))

            axes[0].set_title(cols[0], fontsize=40)  # set column label --> considered epoch
            axes[1].set_title(cols[1], fontsize=40)  # set column label --> considered epoch
            axes[2].set_title(cols[2], fontsize=40)  # set column label --> considered epoch

            image_s = image_t.cpu().numpy()
            mask_s = mask_t.squeeze(0).cpu().numpy()
            pred_s = pred_t.squeeze(0).cpu().numpy()

            image = image_s.transpose(1, 2, 0)
            mask = Image.fromarray(mask_s.astype('uint8')).convert('P')
            pred = Image.fromarray(pred_s.astype('uint8')).convert('P')

            mask.putpalette(pal)
            pred.putpalette(pal)

            axes[0].imshow(image)
            axes[1].imshow(mask)
            axes[2].imshow(pred)

            figpath = os.path.join(fig_output_dir, f"{arch}_{enc_name}_{filename}.png")
            fig.savefig(figpath)
            plt.close()


def export_sample_output_images_comparison(pl_model_list, subset_ds, arch_list, enc_name_list,
                                           device, fig_output_dir, palette, number_samples=None):
    pal = [value for color in palette for value in color]
    number_samples = len(subset_ds) if number_samples is None else number_samples

    random_ids = random.sample(range(len(subset_ds)), number_samples)

    for sample in random_ids:
        cols = ['Image', 'Mask']
        for arch, enc_name in zip(arch_list, enc_name_list):
            cols.append(f"{arch}_{enc_name}")
        fig, axes = plt.subplots(1, len(cols), sharex='row', sharey='row',
                                 subplot_kw={'xticks': [], 'yticks': []},
                                 tight_layout=True, figsize=(len(cols) * 5 + 1, 6))

        for i in range(len(cols)):
            axes[i].set_title(cols[i], fontsize=30)

        batch = subset_ds[sample]
        mask = batch['mask']
        image_original = batch['image']
        image = torch.tensor(image_original).unsqueeze(0).float().to(device)

        mask = Image.fromarray(mask.squeeze(0)).convert('P')
        mask.putpalette(pal)
        pred_list = []
        for pl_model in pl_model_list:
            pred_dl = torch.argmax(pl_model(image), 1)
            pred_dl = Image.fromarray(np.array(pred_dl.squeeze(0).cpu()).astype('uint8')).convert('P')
            pred_dl.putpalette(pal)
            pred_list.append(pred_dl)

        axes[0].imshow(np.array(image_original).transpose(1, 2, 0))
        axes[1].imshow(mask)
        for i in range(len(pred_list)):
            axes[2+i].imshow(pred_list[i])

        filename = subset_ds.filenames[sample]
        figpath = os.path.join(fig_output_dir, f"Comparison_{sample}_{filename}.png")
        plt.tight_layout()
        fig.savefig(figpath)
        plt.close()
