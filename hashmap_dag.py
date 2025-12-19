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
    




