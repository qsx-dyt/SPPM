from .ViCT import build_vict
from .SimMIM import build_simmim


def build_model(args, num_classes, band, depth, heads, is_pretrain=False):
    if is_pretrain:
        model = build_simmim(args, num_classes=num_classes, band=band, depth=depth, heads=heads)
    else:
        model = build_vict(args, num_classes=num_classes, band=band, depth=depth, heads=heads)

    return model
