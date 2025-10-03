import os

import matplotlib.pyplot as plt
import pandas as pd
import shutil
import torch
import numpy as np
import segmentation_models_pytorch as smp
from semseg.paths import get_logs_cmc, get_logs_aag, get_figure_dir
from semseg.util import compute_metrics_all

use_ckpt = False
arch_list = ["unet", "fpn", "deeplabv3plus"]
enc_name_list = ["resnet34"]
subsets = ['valid', 'test']

loggers = [get_logs_cmc, get_logs_aag]
clf_types_short = ["cmc", "aag"]

best_out_dir = os.path.join(get_figure_dir(), 'Best')
worst_out_dir = os.path.join(get_figure_dir(), 'Worst')
os.makedirs(best_out_dir, exist_ok=True)
os.makedirs(worst_out_dir, exist_ok=True)

metrics_dir = os.path.join(get_figure_dir(), 'Metrics')
os.makedirs(metrics_dir, exist_ok=True)

best_worst_idx = 20


def get_metrics_type_list():
    return {
        "cmc": {"Backbone": [], "Decoder": [], "Cortex": [], "Medulla": [], "CapsuleOther": [], "Mean": []},
        "aag": {"Backbone": [], "Decoder": [], "Glomerulus": [], "Artery": [], "Arteriole": [], "Mean": []}
    }


mappers = [
    ["Cortex", "Medulla", "CapsuleOther"],
    ["Glomerulus", "Artery", "Arteriole"],
]


metrics_dice_dict = {
    'valid': get_metrics_type_list(),
    'test': get_metrics_type_list()
}

metrics_recall_dict = {
    'valid': get_metrics_type_list(),
    'test': get_metrics_type_list()
}

metrics_precision_dict = {
    'valid': get_metrics_type_list(),
    'test': get_metrics_type_list()
}

for logger, clf_short, mapper in zip(loggers, clf_types_short, mappers):
    for arch in arch_list:
        for enc_name in enc_name_list:
            for subset in subsets:
                fig_output_dir = os.path.join(logger(arch, enc_name, use_ckpt), f'sample_output_{subset}')
                metrics_torch_out = os.path.join(logger(arch, enc_name, use_ckpt), f'tp_fp_fn_tn_{subset}.pt')
                if os.path.exists(metrics_torch_out):
                    torch_metrics = torch.load(metrics_torch_out)

                    tp, fp, fn, tn = torch_metrics["tp"], torch_metrics["fp"], torch_metrics["fn"], torch_metrics["tn"]

                    metrics_a = compute_metrics_all(tp, fp, fn, tn)

                    dice_a = metrics_a[2, 1:]
                    mean_dice_a = np.mean(dice_a)
                    rounded_dice = np.round(mean_dice_a, decimals=3)
                    metrics_dice = metrics_dice_dict[subset][clf_short]
                    for i in range(len(dice_a)):
                        dice_a_rounded = np.round(dice_a[i], decimals=3)
                        metrics_dice[mapper[i]].append(dice_a_rounded)
                        # metrics_dice[mapper[i]].append(dice_a[i])
                    # metrics_dice["Mean"].append(mean_dice_a)
                    metrics_dice["Mean"].append(rounded_dice)
                    metrics_dice["Backbone"].append(enc_name)
                    metrics_dice["Decoder"].append(arch)

                    recall_a = metrics_a[0, 1:]
                    mean_recall_a = np.mean(recall_a)
                    metrics_recall = metrics_recall_dict[subset][clf_short]
                    for i in range(len(recall_a)):
                        metrics_recall[mapper[i]].append(recall_a[i])
                    metrics_recall["Mean"].append(mean_recall_a)
                    metrics_recall["Backbone"].append(enc_name)
                    metrics_recall["Decoder"].append(arch)

                    precision_a = metrics_a[1, 1:]
                    mean_precision_a = np.mean(precision_a)
                    metrics_precision = metrics_precision_dict[subset][clf_short]
                    for i in range(len(precision_a)):
                        metrics_precision[mapper[i]].append(precision_a[i])
                    metrics_precision["Mean"].append(mean_precision_a)
                    metrics_precision["Backbone"].append(enc_name)
                    metrics_precision["Decoder"].append(arch)

                    if "filename" in torch_metrics:

                        filenames = torch_metrics["filename"]
                        filenames = np.array([f for ff in filenames for f in ff])

                        dices = smp.metrics.f1_score(tp, fp, fn, tn, reduction=None)
                        mean_dice = torch.mean(dices[:, 1:], 1)

                        mean_dice_np = mean_dice.cpu().numpy()

                        plt.figure(figsize=(5, 10))
                        violin_dice = plt.violinplot(mean_dice_np)
                        violin_path = os.path.join(logger(arch, enc_name, use_ckpt), f'dices_violin.png')
                        print(f"Saving violin to {violin_path}")
                        plt.xticks([])
                        plt.title("Violin Plot - Dice Coefficient (Mean)")
                        plt.savefig(violin_path)

                        plt.figure(figsize=(5, 10))
                        boxplot_dice = plt.boxplot(mean_dice_np)
                        boxplot_path = os.path.join(logger(arch, enc_name, use_ckpt), f'dices_boxplot.png')
                        print(f"Saving boxplot to {boxplot_path}")
                        plt.xticks([])
                        plt.title("Box Plot - Dice Coefficient (Mean)")
                        plt.savefig(boxplot_path)


                        print("Filenames    (len):", len(filenames))
                        print("Mean Dice  (shape):", mean_dice.shape)

                        idx_desc = torch.argsort(mean_dice, descending=True)
                        best_idxs = idx_desc[:best_worst_idx]
                        worst_idxs = idx_desc[-best_worst_idx:]
                        best_filenames = filenames[best_idxs]
                        worst_filenames = filenames[worst_idxs]
                        print("Best Segmentation: ", best_filenames)
                        print("Worst Segmentation: ", worst_filenames)
                        for filename in best_filenames:
                            figpath = os.path.join(fig_output_dir, f"{arch}_{enc_name}_{filename}.png")
                            out_path = os.path.join(best_out_dir, f"{clf_short}_{arch}_{enc_name}_{filename}.png")
                            if os.path.exists(figpath):
                                shutil.copy(figpath, out_path)

                        for filename in worst_filenames:
                            figpath = os.path.join(fig_output_dir, f"{arch}_{enc_name}_{filename}.png")
                            out_path = os.path.join(worst_out_dir, f"{clf_short}_{arch}_{enc_name}_{filename}.png")
                            if os.path.exists(figpath):
                                shutil.copy(figpath, out_path)

for subset in metrics_dice_dict:
    for clf_short in metrics_dice_dict[subset]:
        data = metrics_dice_dict[subset][clf_short]
        subset_clf_df = pd.DataFrame(data)
        subset_clf_df.to_csv(os.path.join(metrics_dir, f"{clf_short}_{subset}_dice.csv"), index=False, sep=';')

for subset in metrics_recall_dict:
    for clf_short in metrics_recall_dict[subset]:
        data = metrics_recall_dict[subset][clf_short]
        subset_clf_df = pd.DataFrame(data)
        subset_clf_df.to_csv(os.path.join(metrics_dir, f"{clf_short}_{subset}_recall.csv"), index=False, sep=';')

for subset in metrics_precision_dict:
    for clf_short in metrics_precision_dict[subset]:
        data = metrics_precision_dict[subset][clf_short]
        subset_clf_df = pd.DataFrame(data)
        subset_clf_df.to_csv(os.path.join(metrics_dir, f"{clf_short}_{subset}_precision.csv"), index=False, sep=';')
