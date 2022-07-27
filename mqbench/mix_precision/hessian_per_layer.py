from tkinter.tix import Tree
from typing import Dict

import torch
import numpy as np
from pyhessian import hessian, hessian_vector_product, group_product, orthnormal, normalization

from mqbench.mix_precision.utils import FWDSaverHook, BWDSaverHook, get_actis_grad, hook_model

class hessian_per_layer(hessian):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.full_dataset:
            self.name2param_idx = {}
            param_idx = 0 
            for name, mod in self.model.named_modules():
                if isinstance(mod, (torch.nn.Conv2d, torch.nn.Linear)):
                    while mod.weight is not self.params[param_idx]:
                        param_idx += 1
                    self.name2param_idx[name] = param_idx
            self.hooked = False 
        else:
            self.first_order_grad_dict = {}
            for name, mod in self.model.named_modules():
                if isinstance(mod, (torch.nn.Conv2d, torch.nn.Linear)):
                    self.first_order_grad_dict[name] = mod.weight.grad + 0.

    def layer_eigenvalues(self, maxIter=100, tol=1e-3) -> Dict:
        """
        compute the top_n eigenvalues in one model by layer.
        """
        device = self.device
        max_eigenvalues_dict = {}

        for name, mod in self.model.named_modules():
            if isinstance(mod, (torch.nn.Conv2d, torch.nn.Linear)):
                weight = mod.weight
                eigenvectors = []
                eigenvalue = None
                v = [torch.randn(weight.size()).to(device)]
                v = normalization(v)

                for i in range(maxIter):
                    v = orthnormal(v, eigenvectors)
                    self.model.zero_grad()

                    if self.full_dataset:
                        # todo
                        tmp_eigenvalue, Hv = self.dataloader_hv_product(v)
                    else:
                        first_order_grad = self.first_order_grad_dict[name]
                        Hv = hessian_vector_product(first_order_grad, weight, v)
                        tmp_eigenvalue = group_product(Hv, v).cpu().item()

                    v = normalization(Hv)

                    if eigenvalue is None:
                        eigenvalue = tmp_eigenvalue
                    else:
                        if abs(eigenvalue - tmp_eigenvalue) / (abs(eigenvalue) + 1e-6) < tol:
                            break
                        else:
                            eigenvalue = tmp_eigenvalue
                max_eigenvalues_dict[name] = eigenvalue

        return max_eigenvalues_dict

    def layer_trace(self, maxIter=100, tol=1e-3) -> Dict:
        """
        Compute the trace of hessian in one model by layer.
        """
        device = self.device
        trace_dict = {}
        idx = 0
        for name, mod in self.model.named_modules():
            if isinstance(mod, (torch.nn.Conv2d, torch.nn.Linear)):
                trace_vhv = []
                trace = 0.
                weight = mod.weight
                for i in range(maxIter):
                    self.model.zero_grad()
                    v = torch.randint_like(weight, high=2, device=device)
                    # generate Rademacher random variables
                    v[v == 0] = -1
                    if self.full_dataset:
                        vs = [torch.zeros_like(p) for p in self.params]
                        vs[self.name2param_idx[name]] = v
                        v = vs
                        _, Hv = self.dataloader_hv_product(v)
                    else:
                        first_order_grad = self.first_order_grad_dict[name]
                        Hv = hessian_vector_product(first_order_grad, weight, v)
                    trace_vhv.append(group_product(Hv, v).cpu().item())
                    if abs(np.mean(trace_vhv) - trace) / (trace + 1e-6) < tol:
                        break
                    else:
                        trace = np.mean(trace_vhv)
                trace_dict[name] = trace
                idx += 1

        return trace_dict


class hessian_per_layer_acti(hessian):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.first_order_grad_dict = {}
        if self.full_dataset:
            self.name2acti_idx = {}
            param_idx = 0 
            cnt = 0
            for name, mod in self.model.named_modules():
                if isinstance(mod, (torch.nn.Conv2d, torch.nn.Linear)):
                    while mod.weight is not self.params[param_idx]:
                        param_idx += 1
                    self.name2acti_idx[name] = cnt 
                    cnt += 1
            self.cache_acti_size()
            self.hooked = False
        else:
            data = kwargs['data']
            self.layer_prepare(data[0], data[1])

    def dataloader_hv_product_for_acti(self, v):
        device = self.device
        num_data = 0  # count the number of datum points in the dataloader

        THv = [0. for _ in v]  # accumulate result

        self.model.zero_grad()
       
        if self.hooked is False:        
            self.grad_savers = {}
            self.acti_savers = {}
            self.hooks = []
            for name, mod in self.model.named_modules():
                if isinstance(mod, (torch.nn.Conv2d, torch.nn.Linear)):
                    self.grad_savers[name] = BWDSaverHook()
                    self.acti_savers[name] = FWDSaverHook()
                    mod.register_forward_hook(self.acti_savers[name])
                    mod.register_backward_hook(self.grad_savers[name])
                    self.hooks.append(self.acti_savers[name])
                    self.hooks.append(self.grad_savers[name])
            self.hooked = True
        for inputs, targets in self.data:
            self.model.zero_grad()
            tmp_num_data = inputs.size(0)
            outputs = self.model(inputs.to(device))
            loss = self.criterion(outputs, targets.to(device))
            loss.backward(create_graph=True)
            self.grad_dict = {key: self.grad_savers[key].store_input[0] + 0. for key in self.grad_savers}
            self.acti_dict = {key: self.acti_savers[key].store_input for key in self.acti_savers}
            # grads = [self.grad_savers[key].store_input[0] + 0. for key in self.grad_savers]
            # actis = [self.acti_savers[key].store_input + 0. for key in self.acti_savers]
            actis = [self.acti_dict[name] for name in self.acti_dict] 
            grads = [self.grad_dict[name] for name in self.grad_dict]
            self.model.zero_grad()
            Hv = torch.autograd.grad(grads,
                                     actis,
                                     grad_outputs=v,
                                     only_inputs=True,
                                     retain_graph=False)
            THv = [
                THv1 + Hv1 * float(tmp_num_data) + 0.
                for THv1, Hv1 in zip(THv, Hv)
            ]
            num_data += float(tmp_num_data)
            # outputs = None 
            # loss = None
            # self.grad_dict = {}
            # self.acti_dict = {}

            for hk in self.hooks:
                hk.store_input = None
            torch.cuda.empty_cache()


        THv = [THv1 / float(num_data) for THv1 in THv]
        eigenvalue = group_product(THv, v).cpu().item()
        return eigenvalue, THv
    
    def cache_acti_size(self):
        self.name2size = {}
        _handles = {}
        for name, mod in self.model.named_modules():
            if isinstance(mod, (torch.nn.Conv2d, torch.nn.Linear)):
                hook = FWDSaverHook()
                handle = mod.register_forward_hook(hook)
                _handles[name] = (handle, hook)
        self.model(self.data[0][0].cuda())
        for name in _handles:
            self.name2size[name] = _handles[name][1].store_input.shape
            _handles[name][0].remove()


    def layer_prepare(self, data, target):
        self.grad_savers = {}
        self.acti_savers = {}
        self.model.zero_grad()
        for name, mod in self.model.named_modules():
            if isinstance(mod, (torch.nn.Conv2d, torch.nn.Linear)):
                self.grad_savers[name] = BWDSaverHook()
                self.acti_savers[name] = FWDSaverHook()
                mod.register_forward_hook(self.acti_savers[name])
                mod.register_backward_hook(self.grad_savers[name])
        los = torch.nn.CrossEntropyLoss()
        loss = los(self.model(data.cuda()), target.cuda())
        loss.backward(create_graph=True)
        self.grad_dict = {key: self.grad_savers[key].store_input[0] + 0. for key in self.grad_savers}
        self.acti_dict = {key: self.acti_savers[key].store_input for key in self.acti_savers}

    def layer_eigenvalues(self, maxIter=100, tol=1e-3) -> Dict:
        """
        compute the top_n eigenvalues in one model by layer.
        """
        device = self.device
        max_eigenvalues_dict = {}

        for name, mod in self.model.named_modules():
            if isinstance(mod, (torch.nn.Conv2d, torch.nn.Linear)):
                if self.full_dataset:
                    v = [torch.zeros(self.name2size[_name]).to(device) for _name in self.name2size]
                else:
                    acti = self.acti_dict[name]
                    max_eigenvalues_dict[name] = []
                    first_order_grad = self.grad_dict[name]
                    v = [torch.randn(acti.size()).to(device)]
                v = normalization(v)
                eigenvectors = []
                eigenvalue = None
                
                for i in range(maxIter):
                    v = orthnormal(v, eigenvectors)
                    self.model.zero_grad()

                    if self.full_dataset:
                        v[self.name2acti_idx[name]] = torch.randn(self.name2size[name]).to(device)
                        tmp_eigenvalue, Hv = self.dataloader_hv_product_for_acti(v)
                    else:
                        actis = [self.acti_dict[name] for name in self.acti_dict] 
                        grads = [self.grad_dict[name] for name in self.grad_dict]
                        v = [torch.randn_like(a) for a in actis]
                        Hv = hessian_vector_product(grads, actis, v)
                        Hv = hessian_vector_product(first_order_grad, acti, v)
                        tmp_eigenvalue = group_product(Hv, v).cpu().item()

                    v = normalization(Hv)
                    for i in range(len(Hv)):
                        Hv[i] = None

                    if eigenvalue is None:
                        eigenvalue = tmp_eigenvalue
                    else:
                        if abs(eigenvalue - tmp_eigenvalue) / (abs(eigenvalue) + 1e-6) < tol:
                            max_eigenvalues_dict[name] = eigenvalue
                            break
                        else:
                            eigenvalue = tmp_eigenvalue
                        max_eigenvalues_dict[name] = eigenvalue

        return max_eigenvalues_dict

    def layer_trace(self, maxIter=100, tol=1e-3) -> Dict:
        """
        Compute the trace of hessian in one model by layer.
        """
        device = self.device
        trace_dict = {}
        for name, mod in self.model.named_modules():
            if isinstance(mod, (torch.nn.Conv2d, torch.nn.Linear)):
                trace_vhv = []
                trace = 0.
                if self.full_dataset:
                    v = [torch.zeros(self.name2size[_name]).to(device) for _name in self.name2size]
                else:
                    acti = self.acti_dict[name]
                    first_order_grad = self.grad_dict[name]
                for i in range(maxIter):
                    self.model.zero_grad()
                    if self.full_dataset:
                        vv = torch.randint(size=self.name2size[name], high=2, device=device)
                        vv[vv == 0] = -1 
                        v[self.name2acti_idx[name]] = vv 
                        _, Hv = self.dataloader_hv_product_for_acti(v)
                    else:
                        v = torch.randint_like(acti, high=2, device=device)
                        # generate Rademacher random variables
                        v[v == 0] = -1
                        v = [v]

                        Hv = hessian_vector_product(first_order_grad, acti, v)
                    trace_vhv.append(group_product(Hv, v).cpu().item())
                    if abs(np.mean(trace_vhv) - trace) / (trace + 1e-6) < tol:
                        break
                    else:
                        trace = np.mean(trace_vhv)
                    torch.cuda.empty_cache()
                trace_dict[name] = trace

        return trace_dict