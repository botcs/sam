# SAM - RESEARCH branch
<img src="http://users.itk.ppke.hu/~botcs/sam/img/logo.png" alt="Drawing" width=128/>

## Branch details
This branch is ought keep our recent training scripts and notebooks


## Latest benchmarks

#### Embedding network definition:
```python
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.features = models.squeezenet1_1().features
        self.embedding = nn.Sequential(
            nn.Linear(2048, 512),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(512, 128)
        )

    def forward(self, x):
        x = self.features(x)
        x = nn.functional.adaptive_max_pool2d(x, 2)
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        return self.embedding(x)
```

#### Training results
| |Training set|Test set|
---|---|---
\# Individuals | 467 | 41248
\# Images | 88 | 10000
Accuracy (top-1) | 99.6% | __99.3%__

#### Time complexity

| TASK          | P100          | K80           | __Jetson TX1__  |
| ------------- | -------------:| -------------:| -----------:|
640x480x3 face detection (CPU) | 55.1 ms ± 9.26 µs | 55.1 ms ± 9.26 µs | 149 ms ± 372 µs
640x480 face detection (CPU) | 55.1 ms ± 9.26 µs | 55.1 ms ± 9.26 µs | 123 ms ± 1.17 ms
16x3x96x96 embedding net inference | 3.61 ms ± 141 µs | 5.08 ms ± 6.7 µs | 19.1 ms ± 967 µs
1x3x96x96 embedding net inference | 3.5 ms ± 112 µs  | 3.56 ms ± 133 µs | 10.9 ms ± 358 µs
K-Nearest Neighbour from 10000x128 embeddings| 386 µs ± 112 ns | 648 µs ± 194 ns |2.66 ms ± 14.8 µs


#### Project page
