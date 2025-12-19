# -*- coding: utf-8 -*-
"""
Created on Thu Dec 18 22:08:57 2025

@author: manid
"""



import torch
from modified_functions_tensor_operations import names, tensor_graph
from main_file import display_graph



if __name__ == "__main__":
    #inputs
    starting = torch.tensor([1,2,3,4], dtype=float, requires_grad=True)
    names.add(starting, "starting")
    starting2 = torch.tensor([1,5,3,5], dtype=float, requires_grad=True)
    names.add(starting2, "starting2")
    # im1 = starting * starting
    # #names.add(im1)
    # im2 = im1 * starting2
    # #names.add(im2, "im2")
    # im3 = im2*im1
    # #names.add(im3, "im3")
    # out = im3 * im3
    # #names.add(out, "out")
    out = starting * starting2 + starting
    names.add(out, "out")
    
    display_graph(out)
    