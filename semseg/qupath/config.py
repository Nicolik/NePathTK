import os


def get_all_wsis(wsi_dir, ext, prefix="", limit=None):
    if os.path.exists(wsi_dir):
        all_wsis = [os.path.join(wsi_dir, w) for w in os.listdir(wsi_dir) if w.endswith(ext) and w.startswith(prefix)]
        if limit: all_wsis = all_wsis[:limit]
    else:
        all_wsis = []
    return all_wsis
