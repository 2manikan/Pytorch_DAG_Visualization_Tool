# -*- coding: utf-8 -*-
"""
Created on Thu Dec 11 20:05:13 2025

@author: manid
"""


import torch

class Hashmap:
    def __init__(self):
        self.mapping = {} #pair: (tensor, name)
    
    def get_name(self, t):
        return self.mapping[t]
    
    def add(self, t, name):
        self.mapping[t] = name
    
    def remove(self, t):
        return self.mapping.pop(t)

    def display(self):
        print("-------------------")
        for i in self.mapping.keys():
            print(f'{i} -------> {self.mapping[i]}')
        print("-------------------")





starting = torch.tensor([1,2,3,4], dtype=float, requires_grad=True)
starting2 = torch.tensor([1,5,3,5], dtype=float, requires_grad=True)
im1 = starting * starting
im2 = im1 * starting2
im3 = im2*im1
out = im3 * im3



fringe = [(out.grad_fn, 0)]  # pair: (function, tree display depth)

print("--------------------------")
#DFS (BFS will not work because tree needs to be printed out in order of depth)
while(len(fringe) != 0):
    current_func, current_depth = fringe.pop()
    

    if current_func.__class__.__name__ == 'AccumulateGrad':
        print("--------" * current_depth ,"> LEAF TENSOR")
    elif current_func == None:
        print("--------" * current_depth ,"> LEAF TENSOR")
    else:
       print("--------" * current_depth ,">",current_func)
       for func, _ in current_func.next_functions:
           fringe.append((func, current_depth + 1))
   
        

print("--------------------------")

    
    









# #demo hashmap
# tensor_names = Hashmap()
# r = torch.rand((2,3))
# m = torch.tensor([1,2,3,4])

# tensor_names.add(r, "r")
# tensor_names.add(m, "m")

# tensor_names.display()