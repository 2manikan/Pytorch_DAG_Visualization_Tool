# -*- coding: utf-8 -*-
"""
Created on Thu Dec 18 21:16:03 2025

@author: manid
"""

import torch
from modified_functions_tensor_operations import names, tensor_graph







#inputs
starting = torch.tensor([1,2,3,4], dtype=float, requires_grad=True)
names.add(starting, "starting")
starting2 = torch.tensor([1,5,3,5], dtype=float, requires_grad=True)
names.add(starting2)
im1 = starting * starting
names.add(im1)
im2 = im1 * starting2
names.add(im2, "im2")
im3 = im2*im1
names.add(im3, "im3")
out = im3 * im3
names.add(out, "out")




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

    
    









# #demo hashmap
# tensor_names = Hashmap()
# r = torch.rand((2,3))
# m = torch.tensor([1,2,3,4])

# tensor_names.add(r, "r")
# tensor_names.add(m, "m")

# tensor_names.display()