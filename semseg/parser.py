import argparse


def build_kidney_semseg_parser():
    parser = argparse.ArgumentParser(description='Kidney Segmentation from WSI')
    parser.add_argument('-t', '--tile', type=str, help='path/to/wsi/tiles', required=True)
    parser.add_argument('-o', '--output', type=str, help='path/to/output/masks', required=True)
    parser.add_argument('--n-cpu', type=int, help="Number of workers for data loading", default=0)
    parser.add_argument('--batch-size', type=int, help="Batch size for segmentation model", default=1)
    parser.add_argument('--device', type=str, help="Device to use for processing", default="cuda")
    parser.add_argument('--arch', type=str, help="Architecture of the segmentation model", default="unet")
    parser.add_argument('--enc-name', type=str, help="Encoder of the segmentation model", default="resnet34")
    parser.add_argument('--microscopy-type', type=str, help="Microscopy type", default="paraffin")
    parser.add_argument('--use-tta', action='store_true', help="Whether to use TTA")
    return parser


def build_aggregate_semseg_parser():
    import argparse
    parser = argparse.ArgumentParser(description='Aggregate Tiles to WSI Level')
    parser.add_argument('-w', '--wsi', type=str, help='wsi name without extension', required=True)
    parser.add_argument('-e', '--ext', type=str, help='wsi extension', required=True)
    parser.add_argument('-m', '--masks', type=str, help='path/to/tile/masks', required=True)
    parser.add_argument('--wsi-masks', type=str, help='path/to/wsi/masks', required=True)
    parser.add_argument('-u', '--undersampling', type=int, help='Desired Undersampling', default=16)
    parser.add_argument('-j', '--json-metadata', type=str, help='path/to/json/metadata', required=True)
    parser.add_argument('--segmentation-type', type=str, help='Type of Segmentation [cmc, aag, ifta]', required=True)
    return parser
