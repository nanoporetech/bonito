import torch.nn as nn


def convert_to_2d(model):
    for name, l in model.named_children():
        layer_type = l.__class__.__name__
        if layer_type == 'Conv1d':
            new_layer = nn.Conv2d(l.in_channels, l.out_channels,
                                  (1, l.kernel_size[0]), (1, l.stride[0]),
                                  (0, l.padding[0]), (1, l.dilation[0]),
                                  l.groups, False if l.bias is None else True, l.padding_mode)
            params = l.state_dict()
            params['weight'] = params['weight'].unsqueeze(2)
            new_layer.load_state_dict(params)
            setattr(model, name, new_layer)
        elif layer_type == 'BatchNorm1d':
            new_layer = nn.BatchNorm2d(l.num_features, l.eps)
            new_layer.load_state_dict(l.state_dict())
            new_layer.eval()
            setattr(model, name, new_layer)
        elif layer_type == 'Permute':
            dims_2d = []
            # 1D to 2D: i.e. (2, 0, 1) -> (2, 3, 0, 1)
            for d in l.dims:
                assert(d <= 2)
                dims_2d.append(d)
                if d == 2:
                    dims_2d.append(3)
            l.dims = dims_2d
        else:
            convert_to_2d(l)
