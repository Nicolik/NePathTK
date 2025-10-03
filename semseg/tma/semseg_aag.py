import os
import torch
torch.cuda.empty_cache()
from torch.utils.data import DataLoader, Dataset
import ttach as tta
from semseg.tma.util import get_glomerulus_artery_arteriole_segmenter
from semseg.util import run_segmentation
from semseg.tma.data import TMADatasetInference
from semseg.data import get_inference_augmentation
from semseg.parser import build_kidney_semseg_parser

if __name__ == '__main__':
    parser = build_kidney_semseg_parser()
    args = parser.parse_args()
    n_cpu = args.n_cpu
    batch_size = args.batch_size
    device = args.device

    print(f"Building dataset from {args.tile}")
    pl_model = get_glomerulus_artery_arteriole_segmenter(arch=args.arch, enc_name=args.enc_name, device=device)
    class_names = ["Background", "Glomerulus", "Artery", "Arteriole"]

    dataset = TMADatasetInference(args.tile, transform=get_inference_augmentation())
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=n_cpu)

    os.makedirs(args.output, exist_ok=True)
    multiclass_dir = os.path.join(args.output, 'Multiclass')
    binary_dir = os.path.join(args.output, 'Binary')
    os.makedirs(multiclass_dir, exist_ok=True)
    os.makedirs(binary_dir, exist_ok=True)

    if args.use_tta:
        pl_model = tta.SegmentationTTAWrapper(pl_model, tta.aliases.d4_transform(), merge_mode='mean')
    run_segmentation(pl_model, dataloader, binary_dir, multiclass_dir, class_names, device=device)
