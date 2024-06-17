from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

class PreProcess(Dataset):
    def __init__(self, images, labels, transform):
        super(PreProcess, self).__init__()
        self.images = images
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, x):
        img, label = self.images[x], self.labels[x]
        img = self.transform(img.reshape(img.shape[0], img.shape[1], 1))
        
        
        return img, label