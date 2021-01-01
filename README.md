
# DeepCaps : Going Deeper with Capsule Networks with PyTorch

This implementation is done by referring to the official implementation of DeepCaps by [1], a PyTorch implementation [2] and the official paper at https://arxiv.org/abs/1904.09546. 

## How to Use with your own Custom Dataset
To train on your own custom dataset, simply change the required parameters in `cfg.py` and write your own class to load the dataset in `load_data.py`. Finally, replace line 19 in `train.py` appropriately to point to your custom class. No further changes should be required to train the model. The training can be executed with 
```
python train.py
```

## DeepCaps on FashionMNIST
This implementation was tested on FashionMNIST dataset and it managed to achieve an accuracy of 88% on the testing set. In the official paper however, the model has achieved an accuracy of 94% on this dataset. Below are the results from the training.

#### Loss and Accuracy over 1000 epochs
<img src="readme_images/loss_graph.png" width="800" />  <img src="readme_images/accuracy_graph.png" width="800"/>  

#### Reconstruction Network's Result
Below are the results from the reconstruction network of this model at the end of epoch 0, 500 and 995. The top row are the input images to the model along with their corresponding classes and the bottom row are the images reconstructed back from the final capsules along with the network's class prediction on the given image.

<figure class="image">
  <img src="readme_images/Original_vs_Reconstructed_Epoch_0.png" >
  <div align="center"><figcaption>Epoch 0</figcaption></div>
</figure>

<figure class="image">
  <img src="readme_images/Original_vs_Reconstructed_Epoch_500.png">
  <div align="center"><figcaption>Epoch 500</figcaption></div>
</figure>

<figure class="image">
  <img src="readme_images/Original_vs_Reconstructed_Epoch_995.png">
  <div align="center"><figcaption>Epoch 995</figcaption></div>
</figure>

## Conclusion 
It is evident that the model's loss was steadily decreasing while the accuracy improved over the epochs. The output of the reconstruction network was also improving over the epochs. Perhaps with more adjustments on the learning rate and longer training time, this implementation can achieve a higher accuracy.



## References

[1] https://github.com/brjathu/deepcaps

[2] https://github.com/HopefulRational/DeepCaps-PyTorch
