import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(file_path):
    with open(file_path, 'r') as file:
        data = file.readlines()[1:]  # Skip the header line
    data = []
    labels = []
    for line in data:
        parts = line.strip().split(',')
        data.append([float(x) for x in parts[:-1]])
        labels.append(parts[-1])
    return np.array(data), np.array(labels)

