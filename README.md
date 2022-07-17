# neural_reverse

neural_reverse is a library for reverse engineering with neural networks. 
It currently provides some neural network models for binary code similarity detection, like [Gemini](https://arxiv.org/abs/1708.06525) and [SAFE](https://arxiv.org/abs/1811.05296).


# Installation
The main dependency of neural_reverse is Binary [Ninja](http://binary.ninja), [PyTorch](https://pytorch.org/) and [dgl](https://www.dgl.ai/). 

Binary Ninja provides Python-bindings for binary analysis.
And it is used for extracting instructions and CFGs for binary functions.
To use the python-binding of Binary Ninja, you should get a commercial license 
and install its python api. More details can be found in https://docs.binary.ninja/dev/batch.html.

PyTorch and dgl are open-source libraries used for building neural network models. 
You can just install them with pip/anaconda, just like following:

```
pip install torch dgl
```

