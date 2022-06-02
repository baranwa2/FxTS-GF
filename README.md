# FxTS-GF

This repository implements custom PyTorch optimizers for the Fixed-Time convergent Gradient Flows (FxTS-GF) proposed in our recent AAAI paper (https://arxiv.org/pdf/2112.01363.pdf) titled "Breaking the Convergence Barrier: Optimization via Fixed-Time Convergent Flows". The optimizers can be easily integrated by simply invoking:

```
optimizer = FxTS-Momentum(model.parameters(), lr=learning_rate, momentum=momentum)
```

Here is a quick summary of optimizer's performance for function minimization and training of NNs:

### Function minimization
![Proposed GCN architecture](model.png)

### Training of CNN on MNIST dataset
![Proposed GCN architecture](model.png)

### Training of CNN on CIFAR10 dataset
![Proposed GCN architecture](model.png)
