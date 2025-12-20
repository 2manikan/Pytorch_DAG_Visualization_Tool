# -*- coding: utf-8 -*-
"""
Created on Thu Dec 18 22:08:57 2025

@author: manid
"""



import torch
from modified_functions_tensor_operations import names, tensor_graph, log_dependency
from main_file import display_graph




        

if __name__ == "__main__":
    # #inputs
    # starting = torch.tensor([1,2,3,4], dtype=float, requires_grad=True)
    # names.add(starting, "starting")
    # starting2 = torch.tensor([1,5,3,5], dtype=float, requires_grad=True)
    # names.add(starting2, "starting2")
    # # im1 = starting * starting
    # # #names.add(im1)
    # # im2 = im1 * starting2
    # # #names.add(im2, "im2")
    # # im3 = im2*im1
    # # #names.add(im3, "im3")
    # # out = im3 * im3
    # # #names.add(out, "out")
    # out = starting * starting2 + starting
    # names.add(out, "out")
    
    # display_graph(out)
    
    
    #input and model creation
    inp = torch.randn((4,1)) #batch size 4
    layer1 = torch.nn.Linear(1,1)
    layer2 = torch.nn.Linear(1,1)
    layer3 = torch.nn.Sequential(
        torch.nn.Linear(1,1),
        torch.nn.Linear(1,1)
    )
    
    
    #enable graph logging
    layer1.register_forward_hook(log_dependency)
    layer2.register_forward_hook(log_dependency)
    layer3.register_forward_hook(log_dependency)
    
    
    result = layer3(layer2(layer1(inp)))
    loss = result * 2
    
    
    
    display_graph(loss)
    
    
    
    
    
    
    
    
    
    
    
    