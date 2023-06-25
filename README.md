# Deep Learning: Images

This repo is for learning purposes.

[Transformer Notebook](./mnist_transformer.ipynb):

- an upgrade from the simple [convolutional notebook](./mnist_conv.ipynb)
- use multi-headed attention to encode images as a sequence of patches
- final layer classifier learns to classify based on the encoded representations of the image


[Chest X-Ray Medical Diagnosis Notebook](./xray_conv.ipynb):

- train classifier on medical image data
- deal with class imbalance (very common issue) 
- more practice with convolutions, channels, filters, and dimensions


[Triple Loss Notebook](./mnist_contrastive.ipynb):

- train the model on (anchor, positive, negative) with Triplet Loss for 30 epochs
- then train it with just 5 epochs on Cross-entropy Loss
- the encoder of the embeddings used is that of a Transformer
