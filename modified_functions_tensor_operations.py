# -*- coding: utf-8 -*-
"""
Created on Thu Dec 18 21:12:47 2025

@author: manid
"""

import torch
from hashmap_dag import Tensor_Graph, Hashmap


tensor_graph = Tensor_Graph()
names = Hashmap()



#for ease of operations
def new_mul(self, other):
    result = torch._C._TensorBase.__mul__(self, other) #original multiplication method
    
    #The Tensor Graph can have non-grad tensors as LEAVES ONLY. If requires_grad is manually changed to false,  manual operations must be done to the tree
    #The above is true with the actual DAG. intermediate tensors cannot have requires_grad = False.
    if result.requires_grad == True:
        tensor_graph.add_tensor_edge(result, self)
        tensor_graph.add_tensor_edge(result, other)
    
    
    return result
torch.Tensor.__mul__ = new_mul

def new_add(self, other):
    result = torch._C._TensorBase.__add__(self, other) #original multiplication method
    
    #The Tensor Graph can have non-grad tensors as LEAVES ONLY. If requires_grad is manually changed to false,  manual operations must be done to the tree
    #The above is true with the actual DAG. intermediate tensors cannot have requires_grad = False.
    if result.requires_grad == True:
        tensor_graph.add_tensor_edge(result, self)
        tensor_graph.add_tensor_edge(result, other)
    
    
    return result
torch.Tensor.__add__ = new_add

def new_sub(self, other):
    result = torch._C._TensorBase.__sub__(self, other) #original multiplication method
    
    #The Tensor Graph can have non-grad tensors as LEAVES ONLY. If requires_grad is manually changed to false,  manual operations must be done to the tree
    #The above is true with the actual DAG. intermediate tensors cannot have requires_grad = False.
    if result.requires_grad == True:
        tensor_graph.add_tensor_edge(result, self)
        tensor_graph.add_tensor_edge(result, other)
    
    
    return result
torch.Tensor.__sub__ = new_sub

def new_div(self, other):
    result = torch._C._TensorBase.__div__(self, other) #original multiplication method
    
    #The Tensor Graph can have non-grad tensors as LEAVES ONLY. If requires_grad is manually changed to false,  manual operations must be done to the tree
    #The above is true with the actual DAG. intermediate tensors cannot have requires_grad = False.
    if result.requires_grad == True:
        tensor_graph.add_tensor_edge(result, self)
        tensor_graph.add_tensor_edge(result, other)
    
    
    return result
torch.Tensor.__div__ = new_div

def new_matmul(self, other):
    result = torch._C._TensorBase.__matmul__(self, other) #original multiplication method
    
    #The Tensor Graph can have non-grad tensors as LEAVES ONLY. If requires_grad is manually changed to false,  manual operations must be done to the tree
    #The above is true with the actual DAG. intermediate tensors cannot have requires_grad = False.
    if result.requires_grad == True:
        tensor_graph.add_tensor_edge(result, self)
        tensor_graph.add_tensor_edge(result, other)
    
    
    return result
torch.Tensor.__matmul__ = new_matmul