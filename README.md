# SAM - Face recognition system

Face recognition for identification case study written in Python using [PyTorch](http://pytorch.org/), [OpenCV](https://opencv.org/), [dlib](http://dlib.net/)

In this repository all the source are provided for our:
- Case studies
- Data collection / preparation mechanisms
- Training / evaluation scripts

## Project home
For news, and less tech-related info visit the project's page [PPCU - SAM](http://users.itk.ppke.hu/~botcs/sam/).

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
Training set = __467__ Individuals, __41248__ Images
Test set = __88__ Individuals, __10000__ Images

Performance (Eucledian embedding with margin=1) = 99.6% Train | __99.3% Test__

#### Time complexity

| TASK          | P100          | K80           | __Jetson TX1__  |
| ------------- | -------------:| -------------:| -----------:|
640x480x3 face detection (__dlib__ CPU) | 55.1 ms ± 9.26 µs | 55.1 ms ± 9.26 µs | 149 ms ± 372 µs
640x480 face detection (__dlib__ CPU) | 55.1 ms ± 9.26 µs | 55.1 ms ± 9.26 µs | 123 ms ± 1.17 ms
16x3x96x96 embedding net inference (__PyTorch__ GPU) | 3.61 ms ± 141 µs | 5.08 ms ± 6.7 µs | 19.1 ms ± 967 µs
1x3x96x96 embedding net inference (__PyTorch__ GPU) | 3.5 ms ± 112 µs  | 3.56 ms ± 133 µs | 10.9 ms ± 358 µs
K-Nearest Neighbour from 10000x128 embeddings (__PyTorch__ GPU) | 386 µs ± 112 ns | 648 µs ± 194 ns |2.66 ms ± 14.8 µs


## Usage
### Download repo
```
git clone https://github.com/botcs/ppcu-entry/
```

### Install requirements
```
pip install -r requirements.txt
```

### Usage
Currently `burst_record.py` captures streams from two webcams.

Pressing `h` or `j` will record a small burst of a fixed time-frame (asynchronously, in the background, so the main loop can continue) and with the images will be saved to `h/` and `j/` directory 

later the directory can be renamed to the corresponding ID

`python loadOpenFace.py --database testImg/ --webcam` does all the magic.
It reads from the `testImg` folder - which the user should provide with subfolders containing image captures from the corresponding person.
After loading the images the script infers all the images found in the _database_ and stores the resulting feature vector.
Finally a CV2 frame comes up with the **query** image, which is inferred instantenously and based on L2 distance the 5-nearest neighbour is listed in the database.
From the resulting neighbours' path the direct image ID is removed and only the subfolder name is printed - thus presenting the top 5 **match** in the database by ID.
Best scenario is that a _known_ face is recognised and the top 5 subfolder name, and the query ID is matching.
When an unknown face is presented, then usually the network finds a similar face (I mean similar by human means.. cool), or multiple ones, and rapidly switches the top results. Hopefully the listed **distance is above** a specific threshold, so it could be told that the query face is **not in the database**
