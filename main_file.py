# -*- coding: utf-8 -*-
"""
Created on Thu Dec 18 21:16:03 2025

@author: manid
"""

import torch
from modified_functions_tensor_operations import names, tensor_graph


def display_graph(out): #out is the tensor from which the backward graph is created.
    #display
    fringe = [(out, 0)]  # pair: (function, tree display depth)
    print("--------------------------")
    #DFS (BFS will not work because tree needs to be printed out in order of depth)
    while(len(fringe) != 0):
        current_tensor, current_depth = fringe.pop()
        

        
        if current_tensor.grad_fn.__class__.__name__ == 'AccumulateGrad' or current_tensor.grad_fn == None:
            print("--------" * current_depth ,">", names.mapping[current_tensor], "LEAF TENSOR")
        else:
            print("--------" * current_depth ,">", names.mapping[current_tensor], current_tensor.grad_fn)
        
        inp_tensors = tensor_graph.get_contributing_tensors(current_tensor)
        for i in inp_tensors:
            fringe.append((i, current_depth+1))
        
       
            

    print("--------------------------")










    
    







