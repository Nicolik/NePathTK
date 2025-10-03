import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from semseg.paths import get_logs_cmc, get_logs_aag, get_figure_dir

arch_list = ["deeplabv3plus", "fpn", "unet"]
enc_name_list = ["resnet34"]
epochs = 50

loggers = [get_logs_cmc, get_logs_aag]
clf_types = ["Cortex/Medulla/Capsule", "Glomerulus/Artery/Arteriole"]
clf_types_short = ["cmc", "aag"]

metric_plot_files = []
metric_long_filenames = []

for logger, clf_type, clf_short in zip(loggers, clf_types, clf_types_short):
    for arch in arch_list:
        for enc_name in enc_name_list:
            dd = os.path.join(logger(arch, enc_name), f'{arch}_{enc_name}_{epochs}')
            if os.path.exists(dd):
                versions = os.listdir(dd)
                for version in versions:
                    ddv = os.path.join(dd, version)
                    metrics_csv = os.path.join(ddv, "metrics.csv")
                    metrics_df = pd.read_csv(metrics_csv)

                    plt.figure(figsize=(20, 10))
                    title_label = f"Training Trend (Model: {arch}-{enc_name}, Segmentation: {clf_type})"
                    short_title = f"{clf_short}-{arch}-{enc_name}"
                    plt.title(title_label, fontsize=20)
                    plt.ylabel("Metrics", fontsize=18)
                    plt.xlabel("Epoch", fontsize=18)

                    w_size = 5

                    metrics_df_train = metrics_df[['step', 'epoch', 'train_IoU_step', 'train_loss']].dropna()

                    plt.plot(metrics_df_train['step'], metrics_df_train['train_IoU_step'].rolling(w_size).mean(), label="train_IoU")
                    plt.plot(metrics_df_train['step'], metrics_df_train['train_loss'].rolling(w_size).mean(), label='train_loss')

                    epoch_df = metrics_df[['epoch', 'step']]
                    max_step = max(epoch_df['step'])

                    if 'valid_IoU' in metrics_df.columns:

                        metrics_df_val = metrics_df[['step', 'epoch', 'valid_IoU', 'valid_loss']].dropna()

                        best_val_iou = metrics_df['valid_IoU'].max()
                        best_val_loss = metrics_df['valid_loss'].min()

                        best_val_step_iou = metrics_df_val['step'].iloc[metrics_df_val['valid_IoU'].argmax()]
                        best_val_epoch_iou = metrics_df_val['epoch'].iloc[metrics_df_val['valid_IoU'].argmax()]

                        best_val_step_loss = metrics_df_val['step'].iloc[metrics_df_val['valid_loss'].argmin()]
                        best_val_epoch_loss = metrics_df_val['epoch'].iloc[metrics_df_val['valid_loss'].argmin()]

                        plt.hlines([best_val_iou], 0, max_step, linestyles='--', color='gray', label='best_val_IoU')
                        plt.hlines([best_val_loss], 0, max_step, linestyles='--', color='purple', label='best_val_loss')

                        plt.vlines([best_val_step_iou], best_val_iou, 1, linestyles='--', color='gray')
                        plt.vlines([best_val_step_loss], 0, best_val_loss, linestyles='--', color='purple')

                        label_iou = f"IoU: {best_val_iou:.2f} @ Epoch: {best_val_epoch_iou+1}"
                        label_loss = f"Loss: {best_val_loss:.2f} @ Epoch: {best_val_epoch_loss+1}"
                        # plt.annotate(label_iou,
                        #              (best_val_step_iou, best_val_iou), ha='center', va='center',
                        #              textcoords='offset fontsize', xytext=(0, -2), fontsize=14)
                        # plt.annotate(label_loss,
                        #              (best_val_step_loss, best_val_loss), ha='center', va='center',
                        #              textcoords='offset fontsize', xytext=(0, 2), fontsize=14)

                        plt.scatter(best_val_step_iou, best_val_iou, label=label_iou, c='black', marker='o')
                        plt.scatter(best_val_step_loss, best_val_loss, label=label_loss, c='red', marker='o')
                        plt.plot(metrics_df_val['step'], metrics_df_val['valid_IoU'], label='val_IoU', linestyle='dashed', c='black', marker='*')
                        plt.plot(metrics_df_val['step'], metrics_df_val['valid_loss'], label='val_loss', linestyle='dashed' , c='red', marker='*')

                    plt.ylim([0, 1])
                    plt.xlim([0, max_step])

                    xticks = []
                    xlabels = []
                    epoch = 0
                    filter_by_multiple = 5
                    for epoch in epoch_df['epoch'].unique():
                        if epoch % filter_by_multiple == 0:
                            least_step = min(epoch_df['step'][epoch_df['epoch'] == epoch])
                            xticks.append(least_step)
                            xlabels.append(epoch)
                    xticks.append(max_step)
                    xlabels.append(epoch+1)

                    plt.gca().set_xticks(xticks)
                    plt.gca().set_xticklabels(xlabels, fontsize=16)
                    plt.yticks([i/10 for i in range(11)], fontsize=16)

                    plt.legend(loc='center right', fontsize=16)
                    plt.tight_layout()

                    plot_file = os.path.join(ddv, "metrics.png")
                    plt.savefig(plot_file)
                    plt.close()

                    metric_plot_files.append(plot_file)
                    metric_long_filenames.append(f"{short_title}.png")

out_dir = os.path.join(get_figure_dir(), 'Trends')
os.makedirs(out_dir, exist_ok=True)
for metric_plot_file, metric_dst_file in zip(metric_plot_files, metric_long_filenames):
    dst_path = os.path.join(out_dir, metric_dst_file)
    print(f"Copying {metric_plot_file} to {dst_path}...")
    shutil.copy(metric_plot_file, dst_path)
