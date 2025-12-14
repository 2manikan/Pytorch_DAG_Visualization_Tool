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
        self.count = 0
    
    def get_name(self, t):
        return self.mapping[(t)]
    
    def add(self, t, name = None):
        
        if name != None:
           self.mapping[(t)] = name
        else:
           self.mapping[t] = "unnamed_tensor "+str(self.count)
           self.count += 1
    
    def remove(self, t):
        return self.mapping.pop((t))

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
        if (t1) not in self.edges.keys():
           self.edges[(t1)] = [(t2)]
        else:
           self.edges[(t1)].append((t2)) 
    
    def remove_tensor_edge(self, t1):
        return self.edges.pop((t1))
    
    def get_contributing_tensors(self, t):
        if (t) in self.edges.keys():
           return self.edges[(t)]
        else:
           return []
    
tensor_graph = Tensor_Graph()
names = Hashmap()


#for ease of operations
def new_mul(self, other):
    result = torch._C._TensorBase.__mul__(self, other) #original multiplication method
    
    #The Tensor Graph can have non-grad tensors as LEAVES ONLY. If requires_grad is manually changed to false,  manual operations must be done to the tree
    if result.requires_grad == True:
        tensor_graph.add_tensor_edge(result, self)
        tensor_graph.add_tensor_edge(result, other)
    
    
    return result
torch.Tensor.__mul__ = new_mul






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
    

    
    if current_tensor.grad_fn.__class__.__name__ == 'AccumulateGrad' or current_tensor.grad_fn == None or current_tensor.requires_grad == False:
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