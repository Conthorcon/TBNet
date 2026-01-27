from .SINet.SINet import SINet_ResNet50
from .ACUMEN.clipcod import CLIPCOD
from .TBNet.TBNet import TBNet, BGNet
from .TBNet.CGD import Network
from .BGNet.bgnet import Net
from loguru import logger

def build_segmenter(args):
    if args.model == 'SINet':
        model = SINet_ResNet50()
    elif args.model == 'ACUMEN':
        model = CLIPCOD(args)
    elif args.model == 'TBNet':
        fl = [64, 128, 320, 512]
        model = TBNet(encoder=BGNet(), backbone=Network(fl=fl))
    elif args.model == 'BGNet':
        model = Net()
    backbone = []
    head = []
    for k, v in model.named_parameters():
        if k.startswith('backbone') and 'positional_embedding' not in k:
            backbone.append(v)
        else:
            head.append(v)
    logger.info('Backbone with decay={}, Head={}'.format(len(backbone), len(head)))
    param_list = [{
        'params': backbone,
        'initial_lr': args.lr_multi * args.base_lr
    }, {
        'params': head,
        'initial_lr': args.base_lr
    }]
    return model, param_list