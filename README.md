# Fashion MNIST CNN Classifier

This is a CNN for image classification trained on the FashionMNIST public dataset. It takes an image of clothing and outputs what the name of the article is.

While completing the online course, Deep Learning in Python, I created this project to solidify my understanding of CNN models.

I only ran 1 epoch to cut down on training time, but feel free to run it through more training loops! I used a convolution layer, followed by the ELU activation function for non-linearity, and a pooling layer to reduce dimensionality. Then this is flattened and sent through a linear layer which finally gives me scores that show how likely the image is to be a specific article of clothing. I did not implement Softmax activation, since my goal was to train the model, checking validity with Cross Entropy Loss (which uses Softmax internally).

## Results

Trained for 7 epochs — overall accuracy: **91.0%**

| Class | Precision | Recall |
|---|---|---|
| T-shirt/top | 0.83 | 0.89 |
| Trouser | 0.99 | 0.98 |
| Pullover | 0.80 | 0.91 |
| Dress | 0.90 | 0.92 |
| Coat | 0.87 | 0.84 |
| Sandal | 0.99 | 0.97 |
| Shirt | 0.81 | 0.66 |
| Sneaker | 0.96 | 0.97 |
| Bag | 0.98 | 0.98 |
| Ankle boot | 0.97 | 0.97 |

## Classes

| Class | Label |
|---|---|
| 0 | T-shirt/top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle boot |

## How to run

```bash
pip install torch torchvision torchmetrics
python cnn.py
```
