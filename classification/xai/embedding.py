import os
import pickle

import matplotlib.pyplot as plt
import pandas as pd
import torch
from umap import UMAP
from sklearn.manifold import TSNE
import seaborn as sns
import numpy as np
import random
from PIL import Image, ImageOps

from classification.xai.utils import (
    get_dict_label_rgb_mesc, get_dict_border_rgb_mesc,
    get_dict_label_rgb_sclerosis, get_dict_border_rgb_sclerosis
)
from definitions import ROOT_DIR


def extract_features(model, dl, device=torch.device('cuda'), use_nested_loop=True, image_dir=None):
    features = []
    labels = []
    filenames = []
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(dl):

            img = data["image"].to(device)
            label = data["label"].to(device)
            ext_ids = data["external_id"]

            res = model(img)
            res = torch.squeeze(res)
            res = res.cpu().numpy()

            label = label.cpu().numpy()

            # print("res   shape = ", res.shape)
            # print("label shape = ", label.shape)
            if len(res.shape) > 1:
                features.extend(res)
            else:
                features.append(res)
            labels.extend(label)

            for ext_id in ext_ids:
                # print("ext_id = ", ext_id)
                if use_nested_loop:
                    for ext_i in ext_id:
                        # print("Ext id = ", ext_i)
                        filenames.append(ext_i)
                else:
                    if image_dir:
                        ext_id = os.path.join(image_dir, ext_id)
                    filenames.append(ext_id)

    print("Features  len = {}".format(len(features)))
    print("Labels    len = {}".format(len(labels)))
    print("Filenames len = {}".format(len(filenames)))
    df = pd.DataFrame({
        "Features": features,
        "Labels": labels,
        "Filenames": filenames
    })

    return df


def df_to_saveable(df):
    features = df['Features']
    num_features = features[0].shape[0]
    dict_features = {"feat_{}".format(i): [] for i in range(num_features)}
    for j, feature in enumerate(features):
        for i in range(num_features):
            feat_name = "feat_{}".format(i)
            el = float(features[j][i])
            dict_features[feat_name].append(el)

    return pd.DataFrame({
        **dict_features,
        "Labels": df["Labels"],
        "Filenames": df["Filenames"]
    })


def tsne_plot(data, n_components=2, fig_path=None):
    tsne = TSNE(n_components=n_components)
    X = tsne.fit_transform(np.array(data["Features"].to_list()))
    scatter_plot(X, data, fig_path=fig_path)


def umap_plot(data, n_components=2, fig_path=None):
    umap = UMAP(n_components=n_components)
    X = umap.fit_transform(np.array(data["Features"].to_list()))
    scatter_plot(X, data, fig_path=fig_path)


def scatter_plot(X, data, fig_path=None):
    plt.figure()
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=data["Labels"])
    if fig_path:
        plt.savefig(fig_path)
        plt.close()


def fix_local_filenames(filenames):
    new_filenames = []
    for filename in filenames:
        if filename[0] == '.':
            new_filename = ROOT_DIR + filename[1:]
        else:
            new_filename = filename
        new_filenames.append(new_filename)
    return new_filenames


def change_background(image, color):
    pixels = image.load()  # create the pixel map
    for i in range(image.size[0]):  # for every pixel:
        for j in range(image.size[1]):
            if pixels[i, j] == (0, 0, 0, 255):
                pixels[i, j] = color


def tsne_umap_plot_pretty(feature_csv, fig_path=None, image_path=None, num_images_to_plot=1000, embedding_pkl=None,
                          use_tsne=True, labels_encoding=None, change_back=True, target=None, mesc=True, title=None):
    feature_df = pd.read_csv(feature_csv)

    filenames = feature_df['Filenames']
    print(f"Before fix | filenames[0] = {filenames[0]}")
    filenames = fix_local_filenames(filenames)
    print(f"After  fix | filenames[0] = {filenames[0]}")

    labels = feature_df['Labels']
    features = []
    feat_cols = feature_df.columns
    feat_cols = [f for f in feat_cols if 'feat' in f]
    for feat_col in feat_cols:
        feature = feature_df[feat_col].to_numpy()
        features.append(feature)

    features_np = np.array(features)
    cnn_features = np.transpose(features_np)
    print(f"cnn_features.shape = {cnn_features.shape}")
    print(f"features_np.shape  = {features_np.shape}")

    # for filename, label, feature in zip(filenames, labels, features_np):
    #     print("Filename: ...{:10s} | Label: {:d} | Features: {:.2f} {:.2f} {:.2f} {:.2f} ..."
    #           .format(filename[-10:], label, *feature[:4]))

    if embedding_pkl and os.path.exists(embedding_pkl):
        print(f"Loading embeddings from {embedding_pkl}")
        with open(embedding_pkl, "rb") as f:
            filenames, labels, embedding = pickle.load(f)
    else:
        if len(filenames) > num_images_to_plot:
            sort_order = sorted(random.sample(range(len(filenames)), num_images_to_plot))
            filenames = [filenames[i] for i in sort_order]
            labels = [labels[i] for i in sort_order]
            cnn_features = [cnn_features[i] for i in sort_order]
        X = np.array(cnn_features)
        if use_tsne: embedding = TSNE(n_components=2, learning_rate=150, perplexity=30, angle=0.5).fit_transform(X)
        else: embedding = UMAP(n_components=2, n_neighbors=30).fit_transform(X)
        if embedding_pkl:
            with open(embedding_pkl, "wb") as f:
                print(f"Saving embeddings to {embedding_pkl}")
                pickle.dump((filenames, labels, embedding), f)

    tx, ty = embedding[:, 0], embedding[:, 1]
    tx = (tx - np.min(tx)) / (np.max(tx) - np.min(tx))
    ty = (ty - np.min(ty)) / (np.max(ty) - np.min(ty))

    width = 3000
    height = 3000
    max_dim = 100

    if mesc:
        dict_label_rgb = get_dict_label_rgb_mesc()
        dict_border_rgb = get_dict_border_rgb_mesc()
    else:
        dict_label_rgb = get_dict_label_rgb_sclerosis()
        dict_border_rgb = get_dict_border_rgb_sclerosis()

    print(f"filenames len = {len(filenames)}")
    print(f"labels    len = {len(labels)}")
    print(f"tx        len = {tx.shape[0]}")
    print(f"ty        len = {ty.shape[0]}")

    full_image = Image.new('RGBA', (width, height))
    for filename, label, x, y in zip(filenames, labels, tx, ty):
        tile = Image.open(filename).convert('RGBA')
        border = 75
        if change_back:
            change_background(tile, dict_label_rgb[target][label])
            border = 15
        tile = ImageOps.expand(tile, border=border, fill=dict_border_rgb[target][label])
        # tile = ImageOps.expand(tile, border=12, fill=dict_label_color[label])

        rs = max(1, tile.width / max_dim, tile.height / max_dim)
        # tile = tile.resize((int(tile.width / rs), int(tile.height / rs)), Image.ANTIALIAS)
        tile = tile.resize((int(tile.width / rs), int(tile.height / rs)), Image.Resampling.LANCZOS)
        full_image.paste(tile, (int((width - max_dim) * x), int((height - max_dim) * y)), mask=tile.convert('RGBA'))

    TICKSIZE = 60 # 30
    TITLESIZE = 80 # 40
    LABELSIZE = 70 # 36
    MARKERSIZE = 10 # 5
    plt.figure(figsize=(30, 30))
    plt.rc('xtick', labelsize=TICKSIZE)
    plt.rc('ytick', labelsize=TICKSIZE)
    plt.imshow(full_image)
    emb_text = "tSNE" if use_tsne else "UMAP"
    if title is None:
        title = f"{emb_text} Embedding"
    plt.title(title, fontsize=TITLESIZE)
    for key in labels_encoding:
        if change_back:
            plt.scatter(0, 0, c='#%02x%02x%02x%02x' % dict_label_rgb[target][int(key)], marker="s", label=labels_encoding[key])
        else:
            # plt.scatter(0, 0, color=dict_label_color[int(key)], marker="s", label=labels_encoding[key])
            plt.scatter(0, 0, c='#%02x%02x%02x%02x' % dict_border_rgb[target][int(key)], marker="s", label=labels_encoding[key])
    plt.xlabel(f"{emb_text}-1", fontsize=LABELSIZE)
    plt.ylabel(f"{emb_text}-2", fontsize=LABELSIZE)
    plt.legend(fontsize=LABELSIZE, markerscale=MARKERSIZE, loc='best')
    plt.tight_layout()

    if fig_path:
        print(f"Saving {emb_text} to {fig_path}...")
        plt.savefig(fig_path)

    plt.close()

    if image_path:
        full_image.save(image_path)
