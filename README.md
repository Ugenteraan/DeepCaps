
# Deep-CapsNet

This implementation is done by referring to the official implementation of DeepCaps by [1], a PyTorch implementation [2] and the official paper at https://arxiv.org/abs/1904.09546. 

To train on your own custom dataset, simply change the required parameters in `cfg.py` and write your own class to load the dataset in `load_data.py`. Finally, replace line 19 in `train.py` appropriately to point to your custom class. No further changes should be required to train the model. The training can be executed with `python train.py`.

This implementation was tested on FashionMNIST dataset and it managed to achieve an accuracy of 88%. In the official paper however, the model has achieved an accuracy of 94% on this dataset. Perhaps with more adjustments on the learning rate and longer training time, the same accuracy can be achieved. Below are some of the results from the training.

### Lossess and Accuracies over 1000 epochs
<img src="graphs/loss_graph.png" width="200"/>  <img src="graphs/accuracy_graph" width="200"/>  





References:

[1] https://github.com/brjathu/deepcaps
[2] https://github.com/HopefulRational/DeepCaps-PyTorch
