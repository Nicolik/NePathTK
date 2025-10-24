# python .\classification\train\tma_instance\run.py
import os
import shutil
import sys
import pickle
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

base_abs_path = os.path.abspath('.')
print(f"Absolute path: {base_abs_path}")
sys.path.append(base_abs_path)

from definitions import ROOT_DIR
import classification.gutils.net_adapter as cnn
from classification.gutils.data import ImageDatasetFromDF, build_2d_weighted_dataloader, build_2d_dataloader
from classification.gutils.utils import get_proper_device
from classification.inference.utils import (
    inference, multiclass_roc_curve, plot_confusion_matrix, compute_metrics, save_metrics_to_csv
)
from classification.train.utils import train_loop, loss_plot, save_misclassifications
from classification.train.tma_instance.paths import get_logs_path, get_root_dir, rectify_ext_ids
from classification.xai.embedding import extract_features, df_to_saveable, tsne_umap_plot_pretty
from classification.inference.utils_v2 import compute_curves_v2, plot_curves, default_style_fn
from classification.download import download_tma_classification_model

root_dir = get_root_dir()
print(f"Dataset Dir: {root_dir}")

from classification.train.tma_instance.params import (
    label_mappings, classes_encodings, class_names, exclude_labels, targets, train_configs, val_configs
)
base_dir = os.path.join(root_dir, 'DatasetInstancesTMAClassification')
dataset_csv = os.path.join(base_dir, 'dataset_tma_classification_holdout_mask_fill_merged.csv')
folds = 1
use_crop_data = True

images_dir = os.path.join(base_dir, 'FlatCropDataset')
splits_df = pd.read_csv(dataset_csv, sep=';')
eval_df = splits_df[splits_df['split-0'].isin(['Val', 'Test'])].copy()
eval_df['output_label'] = (eval_df['label'] == 'yes_tma').astype(int)

seed = 42

use_mimickers = False
root_path = ROOT_DIR
subsample = 10
subset_test = 100
train_version = "V1"
logs_path_base = get_logs_path(root_path)

percent_multiplier = 1
ndigits = 3

net_names = ["resnet34", "densenet121", "efficientnetv2-s", "convnext_small", "swin_v2_b"]
compartments = ('Glomerulus', ('Artery', 'Arteriole'))
compartments_names = ('Glomerulus', 'Artery+Arteriole')
compartments_shorts = ('glom', 'art2')

do_embedding = False  # True
do_plot_misclassifications = False  # True
do_plot_correctly_classified = False
sample_incorrect = None
sample_correct = 100

dataset_tma_only_params = {
    'filter_key': 'GT Label',
    'keep_value': ['TMA', ],
    'discard_value': None,
}

dataset_tma_mimicker_params = {
    'filter_key': None,
    'keep_value': None,
    'discard_value': None,
}

params_to_use = dataset_tma_mimicker_params

weights_dir = os.path.join(ROOT_DIR, 'classification', 'weights', 'tma_instance')
os.makedirs(weights_dir, exist_ok=True)
copy_from_weights_dir = False  # Use to retrieve models from public repository
copy_to_weights_dir = False    # Use for future uploads to public repository

for compartment, comp_name, comp_short in zip(compartments, compartments_names, compartments_shorts):
    print(f"Compartment: {compartment}")
    logs_path = os.path.join(logs_path_base, str(compartment))
    os.makedirs(logs_path, exist_ok=True)
    for net_name in net_names:
        for target, label_mapping, classes_encoding, class_name, train_config, val_config in zip(
                    targets, label_mappings, classes_encodings, class_names, train_configs, val_configs
                ):
            print(f"Label Mapping: {label_mapping}")
            for fold in range(folds):
                print(f"Fold: {fold}")
                split_fold = f'split-{fold}'
                td = ImageDatasetFromDF(splits_df, split_fold, 'Train',
                                        transform=train_config.transform, label_mapping=label_mapping,
                                        exclude_labels=exclude_labels, root_dir=None, path_key='crop_path',
                                        compartment=compartment, **params_to_use)
                vd = ImageDatasetFromDF(splits_df, split_fold, 'Val',
                                        transform=train_config.transform, label_mapping=label_mapping,
                                        exclude_labels=exclude_labels, root_dir=None, path_key='crop_path',
                                        compartment=compartment, **params_to_use)
                ed = ImageDatasetFromDF(splits_df, split_fold, 'Test',
                                        transform=train_config.transform, label_mapping=label_mapping,
                                        exclude_labels=exclude_labels, root_dir=None, path_key='crop_path',
                                        compartment=compartment, **params_to_use)

                tdl = build_2d_weighted_dataloader(td, train_config.batch_size, train_config.train)
                vdl = build_2d_dataloader(vd, val_config.batch_size, val_config.train)
                edl = build_2d_dataloader(ed, val_config.batch_size, val_config.train)

                net = cnn.get_architecture(net_name, train_config.num_classes, pretrained=True)
                criterion = nn.CrossEntropyLoss()
                optimizer = optim.Adam(net.parameters(), lr=train_config.learing_rate)

                device = get_proper_device()

                net_fold = os.path.join(logs_path, net_name, f'fold-{fold}')
                os.makedirs(net_fold, exist_ok=True)

                net_name_local = f'{net_name}_{target}_{train_version}.pth'
                net_name_global = f'{net_name}_{target}_{train_version}_{comp_short}_{split_fold}.pth'

                metrics_pkl_local = os.path.join(net_fold, f'metrics_trends_{net_name}_{target}_{train_version}.pkl')
                metrics_pkl_global = os.path.join(weights_dir, f'metrics_trends_{net_name}_{target}_{train_version}_{comp_short}_{split_fold}.pkl')

                net_path_local = os.path.join(net_fold, net_name_local)
                net_path_global = os.path.join(weights_dir, net_name_global)

                if copy_from_weights_dir:
                    if not os.path.exists(net_path_global) or not os.path.exists(metrics_pkl_global):
                        net_path_global, metrics_pkl_global = download_tma_classification_model(comp_short, net_name)
                    shutil.copy(net_path_global, net_path_local)
                    shutil.copy(metrics_pkl_global, metrics_pkl_local)

                if os.path.exists(net_path_local):
                    print(f"Loading model from {net_path_local}")
                    net = torch.load(net_path_local)
                    with open(metrics_pkl_local, "rb") as f:
                        metrics = pickle.load(f)
                else:
                    print("Starting Train")
                    net, metrics = train_loop(tdl, net, net_name, optimizer, criterion, train_config.epochs, device,
                                              vdl=vdl, valid_on_train=False)
                    print(f"Saving Model to {net_path_local}")
                    torch.save(net, net_path_local)

                    with open(metrics_pkl_local, "wb") as f:
                        pickle.dump(metrics, f)

                if copy_to_weights_dir:
                    shutil.copy(net_path_local, net_path_global)
                    shutil.copy(metrics_pkl_local, metrics_pkl_global)
                    continue

                print("Doing plot from metrics")
                metrics_fig_path = os.path.join(net_fold, f'metrics_trends_{net_name}_{target}_{train_version}.png')
                loss_plot(metrics, metrics_fig_path)

                roc_curves_list = []
                pr_curves_list = []

                for subset, eval_dl in zip(['Val', 'Test'], [vdl, edl]):
                    inference_pkl = os.path.join(net_fold, f'{net_name}_{target}_{train_version}_data_{subset}.pth')
                    if os.path.exists(inference_pkl):
                        print(f"[Eval: {subset}] Reading inference data from {inference_pkl}")
                        with open(inference_pkl, "rb") as f:
                            inference_output = pickle.load(f)
                    else:
                        print(f"[Eval: {subset}] Starting Inference")
                        inference_output = inference(eval_dl, net, net_name, device)
                        print(f"[Eval: {subset}] Writing inference data to {inference_pkl}")
                        with open(inference_pkl, "wb") as f:
                            pickle.dump(inference_output, f)

                    output_list, proba_list, label_list, ext_ids = inference_output
                    ext_ids = rectify_ext_ids(ext_ids, images_dir)
                    mapping = dict(zip(ext_ids, output_list))
                    mapped = eval_df['crop_path'].map(mapping)
                    col_name = f'output_{net_name}'
                    if col_name in eval_df.columns:
                        eval_df[col_name] = eval_df[col_name].fillna(mapped)
                    else:
                        eval_df[col_name] = mapped
                    pct_not_na = eval_df[f'output_{net_name}'].notna().mean() * 100
                    print(f"Percentage of non-NA values in 'output_{net_name}': {pct_not_na:.2f}%")
                    metrics = compute_metrics(label_list, output_list, class_names=class_name)
                    print(metrics)

                    if do_plot_misclassifications:
                        misclassifications_dir = os.path.join(net_fold, "sample-classifications")
                        os.makedirs(misclassifications_dir, exist_ok=True)
                        num_classes = train_config.num_classes
                        save_misclassifications(output_list, label_list, ext_ids, num_classes, misclassifications_dir,
                                                classes_encoding=classes_encoding,
                                                plot_correct=do_plot_correctly_classified,
                                                num_correct=sample_correct, num_misclassified=sample_incorrect)

                    filename_roc_path = os.path.join(net_fold, f'ROC_{net_name}_{target}_{train_version}_{subset}.png')
                    auroc, opt_thrs_roc = multiclass_roc_curve(label_list, proba_list, filename_roc_path, model=net_name,
                                                               classes_encoding=classes_encoding, target=target,
                                                               plot_chance_level_pr=False,
                                                               do_pr_curve=False, upscale_factor=1.8, upscale_line=1.4,
                                                               title=f'{subset} ROC {net_name} ({comp_name})',
                                                               plot_subset=[1,], disable_title=True)

                    filename_pr_path = os.path.join(net_fold, f'PR_{net_name}_{target}_{train_version}_{subset}.png')
                    auprc, opt_thrs_pr = multiclass_roc_curve(label_list, proba_list, filename_pr_path, model=net_name,
                                                              classes_encoding=classes_encoding, target=target,
                                                              do_pr_curve=True, upscale_factor=1.8, upscale_line=1.4,
                                                              title=f'{subset} PR  {net_name} ({comp_name})',
                                                              plot_subset=[1,], disable_title=True)

                    roc_curves_subset = compute_curves_v2(
                        label_list, proba_list, task="roc",
                        classes_encoding=classes_encoding,
                        meta={"model": f"{net_name} ({comp_name})", "split": f"{subset}"}
                    )
                    pr_curves_subset = compute_curves_v2(
                        label_list, proba_list, task="pr",
                        classes_encoding=classes_encoding,
                        meta={"model": f"{net_name} ({comp_name})", "split": f"{subset}"}
                    )

                    roc_curves_list.append(roc_curves_subset)
                    pr_curves_list.append(pr_curves_subset)

                    print(f"[Eval: {subset}] AUROC: {auroc}")
                    print(f"[Eval: {subset}] AUPRC: {auprc}")
                    metrics['overall']['auroc_macro'] = auroc['mean']
                    metrics['overall']['auprc_macro'] = auprc['mean']
                    for cls in classes_encoding:
                        metrics['per_class'][classes_encoding[cls]]['auroc'] = auroc[f"{cls}"]
                        metrics['per_class'][classes_encoding[cls]]['auprc'] = auprc[f"{cls}"]

                    filename_cm_path = os.path.join(net_fold, f'CM_{net_name}_{target}_{train_version}_{subset}.png')
                    plot_confusion_matrix(output_list, label_list, class_names=class_name, filename=filename_cm_path,
                                          title=f'Confusion Matrix ({subset}) {net_name}')

                    filename_metric_path = os.path.join(net_fold, f'Metrics_{net_name}_{target}_{train_version}_{subset}.csv')
                    save_metrics_to_csv(metrics, filename=filename_metric_path)

                    if do_embedding:
                        name_embedding_file = f'{net_name}_trained_{target}_{train_version}_{subset}'
                        df_path = os.path.join(net_fold, f'Features_{name_embedding_file}.csv')
                        if not os.path.exists(df_path):
                            df = extract_features(net, eval_dl, use_nested_loop=False)
                            saveable_df = df_to_saveable(df)
                            saveable_df.to_csv(df_path)
                        tsne_path = os.path.join(net_fold, f'tsne_{name_embedding_file}.png')
                        umap_path = os.path.join(net_fold, f'umap_{name_embedding_file}.png')
                        itsne_path = os.path.join(net_fold, f'itsne_{name_embedding_file}.png')
                        iumap_path = os.path.join(net_fold, f'iumap_{name_embedding_file}.png')
                        tsne_pkl = os.path.join(net_fold, f'tsne_{name_embedding_file}.pkl')
                        umap_pkl = os.path.join(net_fold, f'umap_{name_embedding_file}.pkl')
                        tsne_umap_plot_pretty(df_path, tsne_path, use_tsne=True, labels_encoding=classes_encoding,
                                              image_path=itsne_path, target=target, change_back=use_crop_data, mesc=False,
                                              embedding_pkl=tsne_pkl, title=f'{subset} tSNE {net_name} ({comp_name})')
                        tsne_umap_plot_pretty(df_path, umap_path, use_tsne=False, labels_encoding=classes_encoding,
                                              image_path=iumap_path, target=target, change_back=use_crop_data, mesc=False,
                                              embedding_pkl=umap_pkl, title=f'{subset} UMAP {net_name} ({comp_name})')

                roc_curves_path = os.path.join(net_fold, f'ROC_{net_name}_{target}_{train_version}_All_Subsets.png')
                all_curves_roc = [c for subset in roc_curves_list for c in subset]
                plot_curves(all_curves_roc, title=f"ROC {net_name} ({comp_name})", save_path=roc_curves_path,
                            style_fn=default_style_fn, upscale_factor=2.0, upscale_line=1.5,
                            percent_multiplier=percent_multiplier, ndigits=ndigits, disable_title=True, dpi=300)

                pr_curves_path = os.path.join(net_fold, f'PR_{net_name}_{target}_{train_version}_All_Subsets.png')
                all_curves_pr = [c for subset in pr_curves_list for c in subset]
                plot_curves(all_curves_pr, title=f"PR {net_name} ({comp_name})", save_path=pr_curves_path,
                            style_fn=default_style_fn, upscale_factor=2.0, upscale_line=1.5,
                            percent_multiplier=percent_multiplier, ndigits=ndigits, disable_title=True, dpi=300)

eval_csv = os.path.join(logs_path_base, "eval_df.csv")
eval_df.to_csv(eval_csv, sep=';', index=False)

base = (
    eval_df.groupby('Biopsy Number', as_index=False)
           .agg({'Institution':'first', 'GT Label':'first', 'split-0':'first', 'output_label':['sum','mean']})
)
base.columns = ['Biopsy Number','Institution','GT Label','split-0','output-label-sum','output-label-mean']
base['output-label-any'] = (base['output-label-sum'] > 0).astype(int)

# Add per-network aggregates
for net_name in net_names:
    col_out = f'output_{net_name}'
    tmp = (
        eval_df.groupby('Biopsy Number', as_index=False)[col_out]
               .agg(['sum','mean'])
               .reset_index()
    )
    tmp.columns = ['Biopsy Number', f'output-{net_name}-sum', f'output-{net_name}-mean']
    tmp[f'output-{net_name}-any'] = (tmp[f'output-{net_name}-sum'] > 0).astype(int)
    base = base.merge(tmp, on='Biopsy Number', how='left')

aggregated_df = base[
    ['Biopsy Number','Institution','GT Label','split-0',
     'output-label-sum','output-label-mean','output-label-any'] +
    sum(([f'output-{n}-sum', f'output-{n}-mean', f'output-{n}-any'] for n in net_names), [])
]

aggregated_csv = os.path.join(logs_path_base, "aggregated_df.csv")
aggregated_df.to_csv(aggregated_csv, sep=';', index=False)
