from prepare.config import (DOWNSAMPLE_FACTOR_TISSUE, DESIRED_OP_TISSUE,
                            DOWNSAMPLE_FACTOR_CMC, DESIRED_OP_CMC,
                            DOWNSAMPLE_FACTOR_GAA, DESIRED_OP_GAA,
                            DOWNSAMPLE_FACTOR_CMC_CZI, DESIRED_OP_CMC_CZI,
                            DOWNSAMPLE_FACTOR_GAA_CZI, DESIRED_OP_GAA_CZI,
                            DOWNSAMPLE_FACTOR_CMC_20, DOWNSAMPLE_FACTOR_GAA_20,
                            MAGNIFICATION, MAGNIFICATION_20, DESIRED_OP_CMC_VS, DESIRED_OP_GAA_VS,
                            DOWNSAMPLE_FACTOR_CMC_VS, DOWNSAMPLE_FACTOR_GAA_VS)


class BaseTestConfig40(object):
    ext = None
    wsi_dir, wsis = None, None
    qupath_segm_tocopy_dir = None
    qupath_segm_dir = None
    path_to_export = None
    downsample_factor_tissue = DOWNSAMPLE_FACTOR_TISSUE
    desired_op_tissue = DESIRED_OP_TISSUE
    downsample_factor_cmc = DOWNSAMPLE_FACTOR_CMC
    desired_op_cmc = DESIRED_OP_CMC
    downsample_factor_aag = DOWNSAMPLE_FACTOR_GAA
    desired_op_aag = DESIRED_OP_GAA
    magnification = MAGNIFICATION


class BaseTestConfig20(BaseTestConfig40):
    downsample_factor_cmc = DOWNSAMPLE_FACTOR_CMC_20
    downsample_factor_aag = DOWNSAMPLE_FACTOR_GAA_20
    magnification = MAGNIFICATION_20


class BaseTestConfigVivaScope(BaseTestConfig40):
    downsample_factor_cmc = DOWNSAMPLE_FACTOR_CMC_VS
    desired_op_cmc = DESIRED_OP_CMC_VS
    downsample_factor_aag = DOWNSAMPLE_FACTOR_GAA_VS
    desired_op_aag = DESIRED_OP_GAA_VS
    magnification = MAGNIFICATION


class BaseTestConfig80(BaseTestConfig40):
    downsample_factor_cmc = DOWNSAMPLE_FACTOR_CMC_CZI
    desired_op_cmc = DESIRED_OP_CMC_CZI
    downsample_factor_aag = DOWNSAMPLE_FACTOR_GAA_CZI
    desired_op_aag = DESIRED_OP_GAA_CZI
