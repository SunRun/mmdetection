from mmdet.datasets import CocoDataset
from mmdet.datasets.registry import DATASETS


@DATASETS.register_module
class FacetsDataset(CocoDataset):
    CLASSES = ["facet"]
