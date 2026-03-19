# Baseline

def get_model(method, args):
    method = method.lower()

    if method == 'baseline':
        from methods.baseline import Baseline
        return Baseline(args)
    elif method == 'inflora':
        from methods.inflora import InfLoRA  # https://arxiv.org/pdf/2404.00228
        return InfLoRA(args)
    elif method == 'sdlora':
        from methods.sdlora import SDLoRA    # https://arxiv.org/pdf/2501.13198
        return SDLoRA(args)
    elif method == 'cllora':
        from methods.cllora import CLLoRA    # https://arxiv.org/pdf/2505.24816
        return CLLoRA(args)
    elif method == 'ewclora':
        from methods.ewclora import EWCLoRA  # https://arxiv.org/abs/2602.17559
        return EWCLoRA(args)
    else:
        raise ValueError(f"Unknown method: {method}")




