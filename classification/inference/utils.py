import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    roc_curve, precision_recall_curve, auc,
    average_precision_score, confusion_matrix,
    classification_report, ConfusionMatrixDisplay,
    accuracy_score, precision_score, recall_score,
)
from matplotlib import pyplot as plt


def inference(vdl, net, net_name, device, use_softmax=False):

    output_list = []
    label_list = []
    external_ids_list = []
    proba_list = np.zeros((len(vdl.dataset), len(vdl.dataset.classes)))

    net.eval()
    net = net.to(device)

    with torch.no_grad():
        for i, data in enumerate(vdl):
            print(f"Iteration [{(i + 1):3d}/{len(vdl):3d}]", end='\r')
            images = data['image']
            labels = data['label']
            ext_ids = data['external_id']
            batch_size = len(images)

            images = images.to(device)
            # labels = labels.to(device)
            if net_name == 'DistillableVit':
                outputs = net.student(images)
            else: outputs = net(images)
            _, predictions = torch.max(outputs, 1)

            if use_softmax:
                outputs = F.softmax(outputs)
            proba_list[(i*batch_size):((i+1)*batch_size), :] = outputs.cpu().numpy()

            predictions = predictions.cpu().numpy()
            labels = labels.cpu().numpy()

            output_list.extend(predictions)
            label_list.extend(labels)
            external_ids_list.extend(list(ext_ids))

    print(f"Output shape: {len(output_list)}")
    return output_list, proba_list, label_list, external_ids_list


def validate(vdl, net, device, criterion):
    net.eval()
    net = net.to(device)

    losses = np.zeros(len(vdl))
    accs = np.zeros(len(vdl))

    with torch.no_grad():
        for i, data in enumerate(vdl):
            print(f"Iteration [{(i + 1):3d}/{len(vdl):3d}]", end='\r')
            images = data['image']
            labels = data['label']

            images = images.to(device)
            labels = labels.to(device)

            outputs = net(images)

            loss = criterion(outputs, labels)
            _, predictions = torch.max(outputs, 1)

            np_pred = predictions.cpu().numpy()
            np_gt = labels.cpu().numpy()
            batch_size = np_pred.shape[0]
            acc = np.sum(np_pred == np_gt) / batch_size

            losses[i] = loss
            accs[i] = acc

    return {
        'mean_acc': np.mean(accs),
        'std_acc': np.std(accs, ddof=1),
        'mean_loss': np.mean(losses),
        'std_loss': np.std(losses, ddof=1)
    }


def collate_external_ids(vdl, use_nested_loop=True, image_dir=None):
    external_ids_list = []

    with torch.no_grad():
        for i, data in enumerate(vdl):
            print(f"Iteration [{(i + 1):3d}/{len(vdl):3d}]")
            ext_ids = data['external_id']
            for ext_id in ext_ids:
                if use_nested_loop:
                    for ext_i in ext_id:
                        external_ids_list.append(ext_i)
                else:
                    if image_dir:
                        ext_id = os.path.join(image_dir, ext_id)
                    external_ids_list.append(ext_id)
    return external_ids_list


def find_best_th_roc_simple(y, y_hat):
    fpr, tpr, thresholds = roc_curve(y, y_hat)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    return optimal_threshold


def find_best_th_pr(y, y_hat, criterion="min"):
    ths = np.linspace(0.001, 0.999, 198)

    recalls    = np.zeros(ths.shape)
    precisions = np.zeros(ths.shape)

    metric_to_beat = 0
    th_to_beat  = 0
    pre_to_beat = 0
    rec_to_beat = 0
    best_index  = 0
    for i, th in enumerate(ths):
        y_hat_l = y_hat >= th
        CM = confusion_matrix(y, y_hat_l)
        (tn, fp, fn, tp) = CM.ravel()
        if tp + fp > 0:
            precision = tp / (tp + fp)
        else:
            precision = 0
        recall = tp / (tp + fn)
        if criterion == "min":
            metric_pr = min(precision, recall)
        elif criterion in ["f1", "dice"]:
            metric_pr = 2 * (precision * recall) / (precision + recall)
        else:
            metric_pr = precision + recall
        if metric_pr > metric_to_beat:
            metric_to_beat = metric_pr
            th_to_beat = th
            pre_to_beat = precision
            rec_to_beat = recall
            best_index = i
    print(f"Best Threshold = {th_to_beat:.3f}")
    print(f"[Precision = {pre_to_beat:.3f}, Recall = {rec_to_beat:.3f}]")

    diff_pre_rec = abs(pre_to_beat - rec_to_beat)
    if diff_pre_rec > 0.1:
        print(f"Recalls    = {recalls[best_index-10:best_index+10]}")
        print(f"Precisions = {precisions[best_index-10:best_index+10]}")
        print(f"Thresholds = {ths[best_index-10:best_index+10]}")
    return th_to_beat


def binary_calibration_curve(y_test, y_score, path_file, model="cls", target=None,
                             upscale_factor=1.5, upscale_line=1.2, num_bins=10):
    prob_true, prob_pred = calibration_curve(y_test, y_score, n_bins=num_bins)
    plt.figure(figsize=(10, 10))

    curve_name = "Calibration Curve"
    xlabel = "Predicted Probability"
    ylabel = "True Probability"

    plt.plot([0, 1], [0, 1], 'k--', linewidth=2*upscale_line, label="Perfect")
    plt.plot(prob_pred, prob_true, linewidth=3*upscale_line, label=model[:-3])
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.xlabel(xlabel, fontsize=18*upscale_factor)
    plt.ylabel(ylabel, fontsize=18*upscale_factor)
    plt.title(f'({target}) {curve_name}', fontsize=20*upscale_factor)
    plt.legend(loc="lower right", fontsize=16*upscale_factor)
    plt.xticks(fontsize=16*upscale_factor)
    plt.yticks(fontsize=16*upscale_factor)
    plt.tight_layout()
    plt.savefig(path_file)
    plt.close()


def plot_trends(thrs, xx, yy, xlabel, ylabel, target, model, path_file_debug,
                upscale_line=1.2, upscale_factor=1.5):
    plt.figure(figsize=(10, 10))
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    cmpl_thrs = [(1 - t) for t in thrs]
    plt.plot(cmpl_thrs, xx[:-1], linewidth=3*upscale_line, label=xlabel)
    plt.plot(cmpl_thrs, yy[:-1], linewidth=3*upscale_line, label=ylabel)
    plt.plot([0.5, 0.5], [0, 1], 'k--', linewidth=1 * upscale_line)
    plt.xlabel("1 - Thresholds", fontsize=18*upscale_factor)
    plt.ylabel("Metric", fontsize=18*upscale_factor)
    plt.title(f'({target}) Trends — {model[:-3]}', fontsize=20*upscale_factor)
    plt.legend(loc="best", fontsize=16*upscale_factor)
    plt.xticks([0.00, 0.25, 0.50, 0.75, 1.00], fontsize=16*upscale_factor)
    plt.yticks(fontsize=16*upscale_factor)
    plt.tight_layout()
    plt.savefig(path_file_debug)
    plt.close()


def binary_roc_curve(y_test, y_score, path_file, model="cls", target=None,
                     upscale_factor=1.5, upscale_line=1.2, do_pr_curve=False,
                     plot_debug=False, path_file_debug=None, criterion_pr="f1",
                     plot_chance_level_pr=False):
    if do_pr_curve:
        precision, recall, thresholds = precision_recall_curve(y_test, y_score)
        auc_ = average_precision_score(y_test, y_score)
        optimal_threshold = find_best_th_pr(y_test, y_score, criterion=criterion_pr)
        xx, yy = recall, precision
        xlabel = "Recall"
        ylabel = "Precision"
        curve_name = "PR"
        loc = "lower center"
    else:
        fpr, tpr, thresholds = roc_curve(y_test, y_score)
        auc_ = auc(fpr, tpr)
        optimal_threshold = find_best_th_roc_simple(y_test, y_score)
        xx, yy = fpr, tpr
        xlabel = "False Positive Rate"
        ylabel = "True Positive Rate"
        curve_name = "ROC"
        loc = "lower right"

    roc_auc = dict()
    roc_auc["0"] = auc_
    roc_auc["1"] = auc_
    roc_auc["mean"] = auc_

    if plot_debug:
        plot_trends(thresholds, xx, yy, xlabel, ylabel,
                    target, model, path_file_debug,
                    upscale_line=upscale_line, upscale_factor=upscale_factor)

    plt.figure(figsize=(10, 10))
    if do_pr_curve:
        if plot_chance_level_pr:
            pr = sum(y_test) / len(y_test)
            plt.plot([0, 1], [pr, pr], 'k--', linewidth=2*upscale_line, label='Chance Level')
    else:
        # plt.plot([0, 1], [0, 1], 'k--', linewidth=2*upscale_line, label='Chance Level')
        plt.plot([0, 1], [0, 1], 'k--', linewidth=2*upscale_line)
    plt.plot(xx, yy, linewidth=3*upscale_line, label=f'{target} (auc: {auc_*100:.1f})')
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel(xlabel, fontsize=18*upscale_factor)
    plt.ylabel(ylabel, fontsize=18*upscale_factor)
    plt.title(f'({target}) {curve_name} — {model[:-3]}', fontsize=20*upscale_factor)
    plt.legend(loc=loc, fontsize=16*upscale_factor)
    plt.xticks(fontsize=16*upscale_factor)
    plt.yticks(fontsize=16*upscale_factor)
    plt.tight_layout()
    plt.savefig(path_file)
    plt.close()

    return roc_auc, optimal_threshold


def _ohe(y):
    y = np.array(y, dtype=np.int64)
    y_ohe = np.zeros((y.size, y.max() + 1))
    y_ohe[np.arange(y.size), y] = 1
    return y_ohe


def multiclass_roc_curve(y_test, y_score, path_file, model="cls", classes_encoding=None, target=None,
                         upscale_factor=1.5, upscale_line=1.2, do_pr_curve=False, criterion_pr="f1",
                         plot_chance_level_pr=False, title=None, plot_subset=None, use_colors_pr=False,
                         auc_multiplier=1, auc_digits=3, disable_title=False):

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    y_test_ohe = _ohe(y_test)
    n_classes = y_test_ohe.shape[1]
    print(f"y test ohe shape = {y_test_ohe.shape}")

    aucs = []
    optimal_thresholds = []
    random_guesses = []

    for i in range(n_classes):
        if do_pr_curve:
            tpr[str(i)], fpr[str(i)], _ = precision_recall_curve(y_test_ohe[:, i], y_score[:, i])
            auc_ = average_precision_score(y_test_ohe[:, i], y_score[:, i])
            optimal_threshold = find_best_th_pr(y_test_ohe[:, i], y_score[:, i], criterion=criterion_pr)
            random_guess = sum(y_test_ohe[:, i]) / len(y_test_ohe[:, i])
            print(f"Class {i}: random_guess: {random_guess:.2f}, P: {sum(y_test_ohe[:, i])}, N: {len(y_test_ohe[:, i])}")
        else:
            fpr[str(i)], tpr[str(i)], _ = roc_curve(y_test_ohe[:, i], y_score[:, i])
            auc_ = auc(fpr[str(i)], tpr[str(i)])
            optimal_threshold = find_best_th_roc_simple(y_test_ohe[:, i], y_score[:, i])
            random_guess = 0
        optimal_thresholds.append(optimal_threshold)
        aucs.append(auc_)
        roc_auc[str(i)] = auc_
        random_guesses.append(random_guess)
    print(f"Random Guesses: {random_guesses}")
    roc_auc["mean"] = np.mean(aucs)

    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

    plt.figure(figsize=(10, 10))
    auc_name = "AUROC"
    if do_pr_curve:
        auc_name = "AUPRC"
        if plot_chance_level_pr:
            for i in range(n_classes):
                if plot_subset is None or i in plot_subset:
                    clse = classes_encoding[i] if classes_encoding[i] != 'nan_label' else 'nan'
                    pr = random_guesses[i]
                    if use_colors_pr:
                        plt.plot([0, 1], [pr, pr], '--', color=colors[i], linewidth=2*upscale_line)
                    else:
                        plt.plot([0, 1], [pr, pr], 'k--', linewidth=2*upscale_line)
                    # , label=f'{clse} Chance Level')
    else:
        plt.plot([0, 1], [0, 1], 'k--', linewidth=2*upscale_line)
    print(roc_auc)
    for i in range(n_classes):
        if plot_subset is None or i in plot_subset:
            clse = classes_encoding[i] if classes_encoding[i] != 'nan_label' else 'nan'
            print(f"Class {i}: {clse} ({auc_name}: {roc_auc[str(i)]*auc_multiplier:.{auc_digits}f})")
            plt.plot(fpr[str(i)], tpr[str(i)], color=colors[i], linewidth=3*upscale_line,
                     label=f'{clse} ({auc_name}: {roc_auc[str(i)]*auc_multiplier:.{auc_digits}f})')
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    if do_pr_curve:
        xlabel = "Recall"
        ylabel = "Precision"
        curve_name = "PR"
        loc = "lower center"
    else:
        xlabel = "False Positive Rate"
        ylabel = "True Positive Rate"
        curve_name = "ROC"
        loc = "lower right"
    plt.xlabel(xlabel, fontsize=16*upscale_factor)
    plt.ylabel(ylabel, fontsize=16*upscale_factor)
    if not disable_title:
        if title is None:
            title = f'{curve_name} ({target}) {model}'
        plt.title(title, fontsize=16*upscale_factor)
    plt.legend(loc=loc, fontsize=14*upscale_factor)
    plt.xticks(fontsize=14*upscale_factor)
    plt.yticks(fontsize=14*upscale_factor)
    plt.tight_layout()
    plt.savefig(path_file)
    plt.close()
    return roc_auc, optimal_thresholds


def multiclass_classification_report(labels, probs, outputs, thresholds):
    """

    :param labels: 1 x d array-like with ground truth labels
    :param probs: k x d array-like with prediction probabilities
    :param outputs: 1 x d array-like with prediction (argmax or default criterion)
    :param thresholds: k thresholds
    :return: dictionary with classification report
    """

    classif_report = classification_report(labels, outputs, output_dict=True)

    for i, threshold in enumerate(thresholds):
        prob = probs[:,i]
        output = [int(p > threshold) for p in prob]
        label = [int(l == i) for l in labels]
        classif_report_binary = classification_report(label, output, output_dict=True)

        classif_report[f"{i}"] = {
            "precision": classif_report_binary["1"]["precision"],
            "recall": classif_report_binary["1"]["recall"],
            "f1-score": classif_report_binary["1"]["f1-score"],
            "support": classif_report_binary["1"]["support"],
        }

    return classif_report


def plot_confusion_matrix(outputs, labels, class_names=None, normalize=False, cmap='Blues',
                          title='Confusion Matrix', filename=None, fontsize=14):
    """
    Plots a confusion matrix using sklearn and matplotlib.

    Parameters:
    - outputs (array-like): Predicted labels.
    - labels (array-like): Ground-truth labels.
    - class_names (list of str, optional): Names of the classes.
    - normalize (bool): Whether to normalize the confusion matrix.
    - cmap (str): Colormap used for the matrix.
    - title (str): Title of the plot.
    """
    # Convert tensors to numpy if needed
    outputs = np.asarray(outputs)
    labels = np.asarray(labels)

    # Compute confusion matrix
    cm = confusion_matrix(labels, outputs, normalize='true' if normalize else None)

    # Display using ConfusionMatrixDisplay
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, cmap=cmap, colorbar=False)
    ax.set_title(title, fontsize=fontsize + 2)
    ax.set_xlabel('Predicted label', fontsize=fontsize + 2)
    ax.set_ylabel('True label', fontsize=fontsize + 2)
    ax.grid(False)
    ax.tick_params(axis='both', labelsize=fontsize)
    for text in disp.text_.ravel():
        text.set_fontsize(fontsize)
    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename)
    else:
        plt.show()
    plt.close()


def compute_metrics(outputs, labels, class_names=None):
    """
    Computes accuracy, precision, recall, and specificity.

    Parameters:
    - outputs (array-like): Predicted labels.
    - labels (array-like): Ground-truth labels.
    - class_names (list of str, optional): Names of the classes.

    Returns:
    - dict with overall and per-class metrics.
    """
    outputs = np.asarray(outputs)
    labels = np.asarray(labels)
    cm = confusion_matrix(labels, outputs)
    num_classes = cm.shape[0]

    # Compute overall metrics
    accuracy = accuracy_score(labels, outputs)
    precision_macro = precision_score(labels, outputs, average='macro', zero_division=0)
    recall_macro = recall_score(labels, outputs, average='macro', zero_division=0)

    # Compute specificity per class
    specificity_per_class = []
    for i in range(num_classes):
        tn = np.sum(np.delete(np.delete(cm, i, axis=0), i, axis=1))
        fp = np.sum(np.delete(cm, i, axis=0)[:, i])
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        specificity_per_class.append(specificity)

    specificity_macro = np.mean(specificity_per_class)

    # Compute precision and recall per class
    precision_per_class = precision_score(labels, outputs, average=None, zero_division=0)
    recall_per_class = recall_score(labels, outputs, average=None, zero_division=0)

    results = {
        'overall': {
            'accuracy': accuracy,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'specificity_macro': specificity_macro
        },
        'per_class': {}
    }

    # Assign per-class metrics
    for i in range(num_classes):
        class_label = class_names[i] if class_names else str(i)
        results['per_class'][class_label] = {
            'precision': precision_per_class[i],
            'recall': recall_per_class[i],
            'specificity': specificity_per_class[i]
        }

    return results


def save_metrics_to_csv(metrics, filename='metrics.csv', ndgits=4):
    """
    Saves the metrics dictionary (from compute_metrics) to a CSV file.

    Parameters:
    - metrics (dict): Dictionary returned by compute_metrics().
    - filename (str): Output CSV file path.
    """
    rows = []

    # Add overall row
    overall = metrics['overall']
    rows.append({
        'class': 'OVERALL',
        'accuracy': round(overall['accuracy'], ndgits),
        'precision': round(overall['precision_macro'], ndgits),
        'recall': round(overall['recall_macro'], ndgits),
        'specificity': round(overall['specificity_macro'], ndgits),
        'auroc': round(overall['auroc_macro'], ndgits),
        'auprc': round(overall['auprc_macro'], ndgits),
    })

    # Add per-class rows
    for cls, vals in metrics['per_class'].items():
        rows.append({
            'class': cls,
            'accuracy': '',
            'precision': round(vals['precision'], ndgits),
            'recall': round(vals['recall'], ndgits),
            'specificity': round(vals['specificity'], ndgits),
            'auroc': round(vals['auroc'], ndgits),
            'auprc': round(vals['auprc'], ndgits),
        })

    df = pd.DataFrame(rows)
    df.to_csv(filename, index=False, sep=';')
    print(f"Compact metrics saved to {filename}")
