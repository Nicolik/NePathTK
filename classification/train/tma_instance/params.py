from classification.train.mesc.config import (
    GlomeruliTrainConfigResizeAutoAugment, GlomeruliTestConfigResize,
)

label_mapping_two_class = {
    0: ['inconclusive', 'no'],
    1: ['yes_tma'],
}
classes_encoding_two_class = {
    0: 'No',
    1: 'TMA',
}
class_names_two_class = [
    'No',
    'TMA'
]
label_mapping_three_class = {
    0: ['no_pathology', 'other_pathology'],
    1: ['collapsed'],
    2: ['globally_sclerosed']
}

label_mappings = [label_mapping_two_class, ]
classes_encodings = [classes_encoding_two_class, ]
class_names = [class_names_two_class, ]
exclude_labels = []
targets = ['two-class', ]

train_configs = [GlomeruliTrainConfigResizeAutoAugment(), ]
val_configs = [GlomeruliTestConfigResize(), ]
