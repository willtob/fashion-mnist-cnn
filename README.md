# Fashion MNIST CNN Classifier

This is a CNN for image classification trained on the FashionMNIST public dataset. It takes an image of clothing and outputs what the name of the article is.

While completing the online course, Deep Learning in Python, I created this project to solidify my understanding of CNN models.

I only ran 1 epoch to cut down on training time, but feel free to run it through more training loops! I used a convolution layer, followed by the ELU activation function for non-linearity, and a pooling layer to reduce dimensionality. Then this is flattened and sent through a linear layer which finally gives me scores that show how likely the image is to be a specific article of clothing. I did not implement Softmax activation, since my goal was to train the model, checking validity with Cross Entropy Loss (which uses Softmax internally).

## How to run

```bash
pip install torch torchvision torchmetrics
python cnn.py
```
