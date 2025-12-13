# -*- coding: utf-8 -*-
"""
Created on Thu Dec 11 20:05:13 2025

@author: manid
"""


import torch


#for tensor names
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


#for tensor structure and storage
class Tensor_Graph:
    def __init__(self):
        self.edges = {} #(result_tensor_id:[original tensor id's])
    
    def add_tensor_edge(self, t1, t2):
        if id(t1) not in self.edges.keys():
           self.edges[id(t1)] = [id(t2)]
        else:
           self.edges[id(t1)].append(id(t2)) 
    
    def remove_tensor_edge(self, t1):
        return self.edges.pop(id(t1))
    
tensor_graph = Tensor_Graph()



#for ease of operations
original_multiplication = torch.Tensor.__mul__  
def new_mul(self, other):
    result = original_multiplication(self, other)
    tensor_graph.add_tensor_edge(result, self)
    tensor_graph.add_tensor_edge(result, other)
    return result
torch.Tensor.__mul__ = new_mul






#inputs
starting = torch.tensor([1,2,3,4], dtype=float, requires_grad=True)
starting2 = torch.tensor([1,5,3,5], dtype=float, requires_grad=True)
im1 = starting * starting
im2 = im1 * starting2
im3 = im2*im1
out = im3 * im3

print(tensor_graph.edges)





#display

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