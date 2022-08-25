# %% Import packages

import random
import torch

from torch.utils.data.sampler import Sampler

from dmcl_examples.experiments.mlp.fashion_mnist.setting5.mcmc.load_datasets import training_dataset

# %% Function for splitting each class of data points into chunks

def chunk(indices, chunk_size):
    return torch.split(torch.tensor(indices), chunk_size)

# %%

dataset = training_dataset
batch_size = 500

num_points = len(dataset)
num_classes = len(dataset.y[0])

num_batches = num_points // batch_size

label_argmax = torch.argmax(dataset.y, axis=1)

class_indices = {i: [] for i in range(num_classes)}

for i in range(num_points):
    class_indices[label_argmax[i].item()].append(i)

# [len(indices[i]) for i in range(10)]

class_props = [len(class_indices[i]) / num_points for i in range(num_classes)]

class_num_batch_points = [int(class_props[i]*batch_size) for i in range(num_classes)]

# %%

for i in range(num_classes):
    random.shuffle(class_indices[i])

class_batch_indices = [chunk(class_indices[i], class_num_batch_points[i]) for i in range(num_classes)]

# [len(class_batch_indices[i]) for i in range(num_classes)]

# [(len(class_batch_indices[i][0]), len(class_batch_indices[i][-1])) for i in range(num_classes)]

# class_batch_indices[0] = class_batch_indices[0][0:111] + (class_batch_indices[0][120], )

# class_batch_indices[0][:-2] + (class_batch_indices[0][-1],)

for i in range(num_classes):
    num_chunks = len(class_batch_indices[i])
    last_batch_size = len(class_batch_indices[i][-1])

    if num_chunks > num_batches:
        class_batch_indices[i] = class_batch_indices[i][:num_batches]
    elif num_chunks == num_batches:
        if last_batch_size < class_num_batch_points[i]:
            repl_class_batch_indices = random.sample(class_indices[i], class_num_batch_points[i]-last_batch_size)

            class_batch_indices[0] = \
                class_batch_indices[0][0:-1] + \
                (torch.cat([class_batch_indices[0][-1], torch.tensor(repl_class_batch_indices)]),)
    elif num_chunks < num_batches:
        num_repl_batches = num_batches - num_chunks

        repl_class_batch_indices = random.sample(
            class_indices[i], (num_repl_batches+1)*class_num_batch_points[i]-last_batch_size
        )

        if last_batch_size == class_num_batch_points[i]:
            class_batch_indices[i] = class_batch_indices[i] + chunk(repl_class_batch_indices, class_num_batch_points[i])
        elif last_batch_size < class_num_batch_points[i]:
            repl_chunks = chunk(repl_class_batch_indices, class_num_batch_points[i])

            class_batch_indices[i] = \
                class_batch_indices[i][0:-1] + \
                (torch.cat([class_batch_indices[i][-1], repl_chunks[-1]]),) + \
                repl_chunks[0:-1]

# %%

batches = []

for j in range(1): # range(num_batches):
    batch = []
    for i in range(num_classes):
        print(len(class_batch_indices[i][j].tolist()))
        batch.extend(class_batch_indices[i][j].tolist())
    batches.append(batch)

for batch in batches:
    random.shuffle(batch)
