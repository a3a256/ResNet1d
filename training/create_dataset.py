import numpy as np
from datasets import load_dataset

def generate_dataset():

    dataset = load_dataset('Falah/Alzheimer_MRI', split='train')
    test = load_dataset('Falah/Alzheimer_MRI', split='test')

    train_images = []
    train_labels = []

    test_images = []
    test_labels = []

    for i in dataset:
        train_images += [np.array(i["image"])]
        train_labels += [i["label"]]

    for i in test:
        test_images += [np.array(i["image"])]
        test_labels += [i["label"]]
        
    train_images = np.array(train_images)
    train_labels = np.array(train_labels)

    test_images = np.array(test_images)
    test_labels = np.array(test_labels)

    return train_images, train_labels, test_images, test_labels