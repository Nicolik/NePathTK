import os.path
from collections import defaultdict
import matplotlib.pyplot as plt
import torchvision


def show_image_montage(dataset, class_names=None, n_rows_per_class=2, n_cols=8, save_dir=None, save_name=None):
    label_to_images = defaultdict(list)

    # Group image indices by class
    for idx in range(len(dataset)):
        label = dataset.targets[idx]
        label_to_images[label].append(idx)

    # Build the grid
    all_images = []
    all_titles = []
    all_ext_ids = []

    labels = sorted(list(label_to_images.keys()))

    for label in labels:
        indices = label_to_images[label]
        selected = indices[:n_rows_per_class * n_cols]  # sample max needed
        for i in selected:
            sample = dataset[i]
            img = sample["image"]
            ext_id = sample["external_id"]
            all_images.append(img)
            all_titles.append(str(label) if class_names is None else class_names[label])
            all_ext_ids.append(ext_id)

    # Make a grid
    grid = torchvision.utils.make_grid(all_images, nrow=n_cols, padding=2, normalize=True)

    print(all_titles)
    print(all_ext_ids)

    print(f"Ext ids: {len(all_ext_ids)}, unique: {len(set(all_ext_ids))}")

    plt.figure(figsize=(n_cols * 2, len(label_to_images) * n_rows_per_class * 2))
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis("off")
    plt.tight_layout(pad=0)

    if save_dir:
        filename = os.path.join(save_dir, f"montage_{save_name}.png")
        plt.savefig(filename)
    else:
        plt.show()
