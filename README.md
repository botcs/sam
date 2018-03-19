# Face recognition (Under development)

Face recognition for identification case study based on [OpenFace](http://cmusatyalab.github.io/openface/) written in Python using PyTorch, OpenCV, dlib

### Project home
http://users.itk.ppke.hu/~botcs/sam/

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
