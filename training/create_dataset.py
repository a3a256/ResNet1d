import numpy as np
from datasets import load_dataset

def generate_dataset():

    dataset = load_dataset('Falah/Alzheimer_MRI', split='train')
    test = load_dataset('Falah/Alzheimer_MRI', split='test')

    images = []
    labels = []

    for i in dataset:
        images += [np.array(i["image"])]
        labels += [i["label"]]
        
    images = np.array(images)
    labels = np.array(labels)

    return images, labels