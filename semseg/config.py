from detection.qupath.defs import PathMESCnn


class SemSegConfig(object):
    batch_size = 1
    n_cpu = 0
    use_tta = False
    microscopy_type = "paraffin"


class SemSegMMConfig(SemSegConfig):
    semseg_cmc = PathMESCnn.SEMSEG_CMC_MM
    semseg_aag = PathMESCnn.SEMSEG_AAG_MM
    semseg_ifta = PathMESCnn.SEMSEG_IFTA_MM


class SemSegVivascopeMMConfig(SemSegMMConfig):
    microscopy_type = "vivascope"


class SemSegSMPConfig(SemSegConfig):
    semseg_cmc = PathMESCnn.SEMSEG_CMC
    semseg_aag = PathMESCnn.SEMSEG_AAG


class DeepLabV3Config(SemSegSMPConfig):
    arch = "deeplabv3plus"
    enc_name = "resnet34"


class UNetConfig(SemSegSMPConfig):
    arch = "unet"
    enc_name = "resnet34"


class SegNetConfig(SemSegSMPConfig):
    arch = "segnet"
    enc_name = "resnet34"


class M2FConfig(SemSegMMConfig):
    arch = "mask2former"
    enc_name = ""


class M2FVivaScopeConfig(SemSegVivascopeMMConfig):
    arch = "mask2former"
    enc_name = ""


class M2FTTAConfig(M2FConfig):
    use_tta = True
