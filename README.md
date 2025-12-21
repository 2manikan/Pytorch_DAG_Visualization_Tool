# Pytorch_DAG_Visualization_Tool <br>

See the story your tensors tell. <br>
A lightweight, non-invasive utility to automatically track and visualize the dependency graph of your PyTorch tensors. No instrumentation, no complicated setup—just name your tensors and watch the graph build itself. <br> <br>

# Why?
This tool gives you the map to debug pytorch tensor dependencies. <br>
It’s for developers, researchers, and students who want to: <br>
- Debug gradient flow in pytorch models. <br>
- Understand the data lineage of any tensor. <br>
- Teach or explain PyTorch’s computational graph visually. <br>

# How It Works

For the standard tensor operations as shown: <br>
<img width="302" height="168" alt="tensor_math" src="https://github.com/user-attachments/assets/76da7939-cee8-42f4-8cfb-e37e317ae478" />  <br> <br>
The Pytorch Directed Acyclic Graph (DAG) is as follows: <br>

<img width="542" height="447" alt="tensor_dependencies" src="https://github.com/user-attachments/assets/f69ad044-77f5-4737-be94-b3dbaac00672" /> <br> <br>

The utility will output the following: <br>
<img width="742" height="231" alt="code_result" src="https://github.com/user-attachments/assets/73abe5e8-36b6-41bc-aab1-bc4e97ebd744" /> <br> <br>

To use the tool: <br> <br>
1. Name your tensors (optional) with a simple names.add(tensor, "name"). <br>
2. Perform operations as usual—basic ops (mul, add, matmul, etc.) are automatically tracked. For pytorch models, please add the line "module.register_forward_hook(log_dependency)" to enable graph tracing. <br>
3. Visualize from any tensor with display_graph(tensor) to see a clean, indented tree of its dependencies. <br>
Please see the demonstration file for more information on syntax. <br> <br>

# Project Structure <br>
Pytorch_DAG_Visualization_Tool <br>
├── hashmap_dag.py           # Core graph & hashmap structures <br>
├── modified_functions_tensor_operations.py  # Automatic operation tracking <br>
├── main_file.py             # Visualization walker (display_graph) <br>
└── demo.py                  # Example usage (run this first!) <br>
No installation required. Clone the repo, run demo.py, and see it work. The commands in demo.py can be used in any file you wish afterward. <br> <br>

# What’s Next? <br>
- Extend operation coverage (e.g., custom autograd Functions). <br>
- Packaging refinement

Contributions, ideas, and feedback are warmly welcomed. Thank you!
