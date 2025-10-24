import time
import os
import shutil
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import pandas as pd

from classification.inference.utils import validate


def train_loop(tdl, net, net_name, optimizer, criterion, epochs, device, vdl=None, valid_on_train=False, use_simple=False):
    net = net.to(device)
    net.train()

    losses_mean_train = np.zeros(epochs)
    losses_std_train = np.zeros(epochs)
    losses_mean_val = np.zeros(epochs)
    losses_std_val = np.zeros(epochs)
    accs_mean_train = np.zeros(epochs)
    accs_std_train = np.zeros(epochs)
    accs_mean_val = np.zeros(epochs)
    accs_std_val = np.zeros(epochs)

    writer = SummaryWriter()

    for epoch in range(epochs):  # loop over the dataset multiple times
        print(f"Training Epoch: {epoch}")
        start_time = time.time()
        # running_loss = 0.0

        for i, data in enumerate(tqdm(tdl), 0):
            if use_simple:
                inputs, labels = data
            else:
                inputs = data['image']
                labels = data['label']

            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            if net_name == "DistillableVit":
                loss = net(inputs, labels)
            else:
                outputs = net(inputs)
                loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            # with torch.no_grad():
            #     running_loss += loss.item()
            #     if i % 10 == 9:  # print every 10 mini-batches
            #         print('[%d, %5d] loss: %.6f' %
            #               (epoch + 1, i + 1, running_loss / 2000))
            #         running_loss = 0.0

        elapsed_time_train_epoch = time.time() - start_time

        if vdl is not None:
            if net_name == "DistillableVit":
                metrics_val = validate(vdl, net.student, device, criterion)
            else:
                metrics_val = validate(vdl, net, device, criterion)
            losses_mean_val[epoch] = metrics_val['mean_loss']
            losses_std_val[epoch] = metrics_val['std_loss']
            accs_mean_val[epoch] = metrics_val['mean_acc']
            accs_std_val[epoch] = metrics_val['std_acc']
            elapsed_time_validate_vdl = time.time() - start_time

            writer.add_scalar("Loss/val", metrics_val["mean_loss"], epoch)

        if valid_on_train:
            if net_name == 'DistillableVit':
                metrics_train = validate(tdl, net.student, device, criterion)
            else:
                metrics_train = validate(tdl, net, device, criterion)
            losses_mean_train[epoch] = metrics_train['mean_loss']
            losses_std_train[epoch] = metrics_train['std_loss']
            accs_mean_train[epoch] = metrics_train['mean_acc']
            accs_std_train[epoch] = metrics_train['std_acc']
            elapsed_time_validate_tdl = time.time() - start_time
        print(f"[{(epoch + 1):03d}/{epochs:03d}] ", end="")
        if vdl is not None and valid_on_train:
            print(f"TAcc = {metrics_train['mean_acc']*100:4.1f} ± {metrics_train['std_acc']*100:4.1f} | "
                  f"VAcc = {metrics_val['mean_acc']*100:4.1f} ± {metrics_val['std_acc']*100:4.1f} | "
                  f"TLoss = {metrics_train['mean_loss']:5.3f} ± {metrics_train['std_loss']:5.3f} | "
                  f"VLoss = {metrics_val['mean_acc']:5.3f} ± {metrics_val['std_acc']:5.3f} | ", end='')
            print(f"Tr = {elapsed_time_train_epoch:6.2f} [s] | "
                  f"Vdl = {elapsed_time_validate_vdl:6.2f} [s] | "
                  f"Tdl = {elapsed_time_validate_tdl:6.2f} [s] |")

        # net_path = os.path.join(r'F:\Python_Repos\QIGS\classification\logs_gastric\cnn\holdout', f'{net_name}_epoch{epoch}.pth')
        # torch.save(net, net_path)

    print('Finished Training')
    metrics = {
        'train': {
            'mean_acc': accs_mean_train,
            'std_acc': accs_std_train,
            'mean_loss': losses_mean_train,
            'std_loss': losses_std_train
        },
        'val': {
            'mean_acc': accs_mean_val,
            'std_acc': accs_std_val,
            'mean_loss': losses_mean_val,
            'std_loss': losses_std_val
        }
    }

    writer.flush()
    writer.close()

    return net, metrics


def loss_plot(metrics, fig_path, plot_train=False):
    plt.figure()

    yat = metrics['train']['mean_acc']
    eat = metrics['train']['std_acc']

    yav = metrics['val']['mean_acc']
    eav = metrics['val']['std_acc']

    ylt = metrics['train']['mean_loss']
    elt = metrics['train']['std_loss']

    ylv = metrics['val']['mean_loss']
    elv = metrics['val']['std_loss']

    xx = np.linspace(1, len(ylv), len(ylv))

    # Train/Val Accuracy
    if plot_train:
        tr_acc = 221
        val_acc = 222
        tr_loss = 223
        val_loss = 224
    else:
        val_acc = 121
        val_loss = 122

    if plot_train:
        plt.subplot(tr_acc)
        plt.title("Train Accuracy", fontsize=14)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.errorbar(xx, yat, yerr=eat, fmt='o', color='blue')
        plt.yticks(np.arange(0, 1.2, step=0.2), fontsize=10)
        plt.xticks(fontsize=10)

    plt.subplot(val_acc)
    plt.title("Validation Accuracy", fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.errorbar(xx, yav, yerr=eav, fmt='o', color='red')
    plt.yticks(np.arange(0, 1.2, step=0.2), fontsize=10)
    plt.xticks(fontsize=10)

    # Train/Val Loss
    if plot_train:
        plt.subplot(tr_loss)
        plt.title("Train Loss", fontsize=14)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.errorbar(xx, ylt, yerr=elt, fmt='o', color='blue')
        plt.yticks(np.arange(0, 2.0, step=0.2), fontsize=10)
        plt.xticks(fontsize=10)

    plt.subplot(val_loss)
    plt.title("Validation Loss", fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.errorbar(xx, ylv, yerr=elv, fmt='o', color='red')
    plt.yticks(np.arange(0, 2.0, step=0.2), fontsize=10)
    plt.xticks(fontsize=10)

    plt.tight_layout()
    plt.savefig(fig_path)


def save_misclassifications(output_list, label_list, ext_ids, num_classes, output_dir,
                            classes_encoding=None, plot_correct=False, num_misclassified=None, num_correct=None):
    output_array = np.array(output_list)
    label_array = np.array(label_list)
    ext_ids_array = np.array(ext_ids)

    # Handle misclassified samples
    for true_label in range(num_classes):
        for predicted_label in range(num_classes):
            if true_label != predicted_label:
                indices = np.where((label_array == true_label) & (output_array == predicted_label))[0]

                # Limit to requested number of misclassified samples
                if num_misclassified is not None:
                    indices = indices[:num_misclassified]

                if len(indices) > 0:
                    if classes_encoding is not None:
                        true_label_val = classes_encoding[true_label]
                        pred_label_val = classes_encoding[predicted_label]
                    else:
                        true_label_val = true_label
                        pred_label_val = predicted_label

                    print(f"\nTrue: {true_label_val} → Predicted: {pred_label_val}")
                    misclassifications_subdir = os.path.join(
                        output_dir, f"L-{true_label_val}__P-{pred_label_val}"
                    )
                    os.makedirs(misclassifications_subdir, exist_ok=True)

                    for eid in ext_ids_array[indices]:
                        print(f"  {eid}")
                        filename = os.path.basename(eid)
                        shutil.copy(eid, os.path.join(misclassifications_subdir, filename))

    # Handle correctly classified samples
    if plot_correct:
        for label in range(num_classes):
            indices = np.where((label_array == label) & (output_array == label))[0]

            # Limit to requested number of correct samples
            if num_correct is not None:
                indices = indices[:num_correct]

            if len(indices) > 0:
                label_val = classes_encoding[label] if classes_encoding is not None else label

                print(f"\nCorrectly classified: {label_val}")
                correct_subdir = os.path.join(output_dir, f"Correct-L-{label_val}")
                os.makedirs(correct_subdir, exist_ok=True)

                for eid in ext_ids_array[indices]:
                    print(f"  {eid}")
                    filename = os.path.basename(eid)
                    shutil.copy(eid, os.path.join(correct_subdir, filename))


def filter_metrics_by_class(full_df, cls, overall_cls="overall"):
    """
    Filter metrics for a given class, handling accuracy with overall_cls if missing.

    Args:
        full_df (pd.DataFrame): Input dataframe with columns
            ['target','model','class','accuracy','precision','recall',
             'specificity','auroc','auprc']
        cls (str): class of interest
        overall_cls (str): fallback class for accuracy if missing

    Returns:
        - If 1 unique target: DataFrame with [model + metrics].
        - If >1 unique target: dict[target -> DataFrame].
    """

    result_dict = {}

    print(full_df["target"].unique())

    for target in full_df["target"].unique():
        print("[filter_metrics_by_class] Processing", target)
        df_t = full_df[full_df["target"] == target]

        # rows for class of interest
        df_cls = df_t[df_t["class"] == cls].copy()

        # rows for fallback (overall_cls) if provided
        if overall_cls is not None:
            df_overall = df_t[df_t["class"] == overall_cls].set_index("model")
        else:
            df_overall = pd.DataFrame()

        # align on model
        df_cls = df_cls.set_index("model")

        # always take accuracy from overall_cls if available
        if overall_cls is not None and not df_overall.empty and "accuracy" in df_cls.columns:
            df_cls["accuracy"] = df_overall["accuracy"]

        # reset index and drop 'target' and 'class'
        df_cls = df_cls.reset_index().drop(columns=["target", "class"])

        result_dict[target] = df_cls

    return result_dict


def scale_metrics(df, factor=1.0, operation="multiply", digits=3):
    """
    Scale all metric columns in df by multiplying or dividing by a factor.

    Args:
        df (pd.DataFrame): DataFrame with metric columns
            ['accuracy','precision','recall','specificity','auroc','auprc']
        factor (float): number to multiply/divide with
        operation (str): "multiply" or "divide"
        digits (int): number of decimal digits to round to

    Returns:
        pd.DataFrame: scaled and rounded DataFrame
    """
    metric_cols = ['accuracy', 'precision', 'recall', 'specificity', 'auroc', 'auprc']
    df_scaled = df.copy()

    for col in metric_cols:
        if col in df_scaled.columns:
            if operation == "multiply":
                df_scaled[col] = pd.to_numeric(df_scaled[col]) * factor
            elif operation == "divide":
                df_scaled[col] = pd.to_numeric(df_scaled[col]) / factor
            else:
                raise ValueError("operation must be 'multiply' or 'divide'")

            numeric_col = df_scaled[col].round(digits)
            df_scaled[col] = numeric_col.map(lambda x: f"{x:.{digits}f}" if pd.notna(x) else x)

    return df_scaled
