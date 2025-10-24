import os
import torch

torch.cuda.empty_cache()
from semseg.mm.util import get_ifta_cortex_mmsegmenter, run_segmentation_mm
from semseg.parser import build_kidney_semseg_parser



if __name__ == '__main__':
    parser = build_kidney_semseg_parser()
    args = parser.parse_args()
    n_cpu = args.n_cpu
    batch_size = args.batch_size
    device = args.device
    tile_dir = args.tile

    print(f"Building dataset from {args.tile}")
    mm_model = get_ifta_cortex_mmsegmenter(arch=args.arch, enc_name=args.enc_name, device=device,
                                           use_tta=args.use_tta, microscopy_type=args.microscopy_type)
    class_names = ["Background", "IFTACortex"]

    print(f"Input dir: {tile_dir}")

    os.makedirs(args.output, exist_ok=True)
    multiclass_dir = os.path.join(args.output, 'Multiclass')
    binary_dir = os.path.join(args.output, 'Binary')
    os.makedirs(multiclass_dir, exist_ok=True)
    os.makedirs(binary_dir, exist_ok=True)

    print(f"TTA: {args.use_tta}")
    run_segmentation_mm(mm_model, tile_dir, binary_dir, multiclass_dir, class_names, device=device, batch_size=batch_size)
