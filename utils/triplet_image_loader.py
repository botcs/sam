from PIL import Image
import os
import os.path
from random import choice
import torch.utils.data
import torchvision.transforms as transforms



def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class TripletImageLoader(torch.utils.data.Dataset):
    """
    Extends torch.utils.data.Dataset class for efficient loading of image triplets.
    In the triplet:
      the first instance is the anchor image
      the second instance is the positive image
      the third instance is the negative image

    The positive image should be a non-identical but semantically close image to anchor.
    The negative image should be a semantically negative image to anchor 
      (i.e. photo of another person)

    Args:
        photoID_path_filename: text file that has records of the following 
          photoID, path = 3125235, "/home/user/img_database/.../125226224.jpg"
        nameID_photoIDs_filename: text file that links photoIDs to nameIDs (personal)
          nameID, photoID = 112, 3125235
          nameID, photoID = 112, 3125236

    Loads:
        anchor, positive, negative
    """
    
    def __init__(self, nameID_photoPaths_filename, 
                 transform=None, loader=default_loader, randomized=True, 
                 triplets_per_individual=100):
        
        self.nameID_photoPaths_database = {}
        with open(nameID_photoPaths_filename) as f:
            for line in f.readlines():
                nameID, photoPath = line[:-1].split('\t')
                if self.nameID_photoPaths_database.get(nameID) is None:
                    self.nameID_photoPaths_database[nameID] = [photoPath]
                else:
                    self.nameID_photoPaths_database[nameID].append(photoPath)
        
        self.nameID_photoPaths_database = list(self.nameID_photoPaths_database.items())
        self.transform = transform
        self.loader = loader
        self.__randomized = randomized
        self.triplets_per_individual = triplets_per_individual
        self.seq_triplets = self.getSequentialTriplets()
        self.rand_triplets = self.getRandomTriplets()
            
        if randomized:
            self.triplets = self.rand_triplets
        else:
            self.triplets = self.seq_triplets
        self.num_of_indivs = len(self.nameID_photoPaths_database)
        self.num_of_triplets = self.getNumofTriplets()
        
    
    def Randomize():
        self.triplets = self.rand_triplets
    
    def Sequentialize():
        self.triplets = self.seq_triplets
    
    def getNumofTriplets(self, avg_approx=True):
        if avg_approx:
            total = sum(len(paths) 
                for _, paths in self.nameID_photoPaths_database)

            return (total // len(self.nameID_photoPaths_database)) ** 3
        else:
            raise NotImplementedError
            
    
    def getSequentialTriplets(self):
        '''
        Problem is O(N^3) complexity, so filling a list is really inefficient,
        however generators are basically cannot be indexed, therefore we implement
        random triplet generator: `getRandomTriplets`
        '''
        for anchorNameID, anchorPhotoPaths in self.nameID_photoPaths_database:
            for anchorPhotoPath in anchorPhotoPaths:
                for negativePhotoPath in negativePhotoPaths:
                    for positivePhotoPath in anchorPhotoPaths:
                        if positivePhotoPath == anchorPhotoPath: continue
                        for negativeNameID, negativePhotoPaths in self.nameID_photoPaths_database:
                            if negativeNameID == anchorNameID: continue
                            yield [anchorPhotoPath, positivePhotoPath, negativePhotoPath]
    
    
    def getRandomTriplets(self):
        i = 0
        while i < self.__len__():
            i += 1
            anchorNameID, anchorPhotoPaths = choice(self.nameID_photoPaths_database)
            negativeNameID, negativePhotoPaths = choice(self.nameID_photoPaths_database)
            while negativeNameID == anchorNameID:
                negativeNameID, negativePhotoPaths = choice(self.nameID_photoPaths_database)
            
            anchorPhotoPath = choice(anchorPhotoPaths)
            positivePhotoPath = choice(anchorPhotoPaths)
            negativePhotoPath = choice(negativePhotoPaths)

            yield [anchorPhotoPath, positivePhotoPath, negativePhotoPath]

    def __getitem__(self, index):
        anchorPath, positivePath, negativePath = next(self.triplets)
        anc_img = self.loader(anchorPath)
        pos_img = self.loader(positivePath)
        neg_img = self.loader(negativePath)
        if self.transform is not None:
            anc_img = self.transform(anc_img)
            pos_img = self.transform(pos_img)
            neg_img = self.transform(neg_img)

        return anc_img, pos_img, neg_img

    def __len__(self):
        if self.__randomized:
            return self.num_of_indivs * self.triplets_per_individual
        else:
            return self.num_of_triplets


