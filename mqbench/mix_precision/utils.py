import torch

class FWDSaverHook:
    """
    Forward hook that stores the input and output of a layer/block
    """
    def __init__(self):
        self.store_input = None 

    def __call__(self, module, input_batch, output_batch):
        output_batch.requires_grad_()
        output_batch.retain_grad()
        self.store_input = output_batch

class BWDSaverHook:
    """
    Forward hook that stores the input and output of a layer/block
    """
    def __init__(self):
        self.store_input = None 

    def __call__(self, module, input_batch, output_batch):
        self.store_input = output_batch

def get_actis_grad(grad_savers, acti_savers):
    actis = [acti_savers[key].store_input + 0. for key in acti_savers]
    grads = [grad_savers[key].store_input[0] + 0. for key in grad_savers]
    assert len(actis) == len(grads)
    return actis, grads
    
def hook_model(model):
    grad_savers, acti_savers = {}, {}
    handles = []
    model.zero_grad()
    for name, mod in model.named_modules():
        if isinstance(mod, (torch.nn.Conv2d, torch.nn.Linear)):
            grad_savers[name] = BWDSaverHook()
            acti_savers[name] = FWDSaverHook()
            acti_handle = mod.register_forward_hook(acti_savers[name])
            grad_handle = mod.register_backward_hook(grad_savers[name])
            handles.append(acti_handle)
            handles.append(grad_handle)
    return grad_savers, acti_savers, handles