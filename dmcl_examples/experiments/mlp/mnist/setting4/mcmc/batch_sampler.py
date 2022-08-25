# %% Import packages

import random
import torch

from torch.utils.data.sampler import Sampler

# %% Function for splitting each class of data points into chunks

def chunk(indices, chunk_size):
    return torch.split(torch.tensor(indices), chunk_size)

# %% Define custom batch sampler

class BatchSampler(Sampler):
    def __init__(self, dataset, batch_size):
        self.num_points = len(dataset)
        self.num_classes = len(dataset.y[0])

        self.batch_size = batch_size
        self.num_batches = self.num_points // self.batch_size

        label_argmax = torch.argmax(dataset.y, axis=1)

        self.class_indices = {i: [] for i in range(self.num_classes)}
        for i in range(self.num_points):
            self.class_indices[label_argmax[i].item()].append(i)

        class_props = [len(self.class_indices[i]) / self.num_points for i in range(self.num_classes)]

        self.class_num_batch_points = [int(class_props[i]*self.batch_size) for i in range(self.num_classes)]

    def fill_class_sizes(self):
        sampled_classes = random.choices(range(self.num_classes), k=self.batch_size-sum(self.class_num_batch_points))

        class_num_batch_points = self.class_num_batch_points.copy()
        for i in sampled_classes:
            class_num_batch_points[i] = class_num_batch_points[i] + 1

        return class_num_batch_points

    def __iter__(self):
        for i in range(self.num_classes):
            random.shuffle(self.class_indices[i])

        class_num_batch_points = self.fill_class_sizes()

        class_batch_indices = [chunk(self.class_indices[i], class_num_batch_points[i]) for i in range(self.num_classes)]

        for i in range(self.num_classes):
            num_chunks = len(class_batch_indices[i])
            last_batch_size = len(class_batch_indices[i][-1])

            if num_chunks > self.num_batches:
                class_batch_indices[i] = class_batch_indices[i][:self.num_batches]
            elif num_chunks == self.num_batches:
                if last_batch_size < class_num_batch_points[i]:
                    repl_class_batch_indices = random.sample(
                        self.class_indices[i], class_num_batch_points[i]-last_batch_size
                    )

                    class_batch_indices[i] = \
                        class_batch_indices[i][0:-1] + \
                        (torch.cat([class_batch_indices[i][-1], torch.tensor(repl_class_batch_indices)]),)
            elif num_chunks < self.num_batches:
                num_repl_batches = self.num_batches - num_chunks

                repl_class_batch_indices = random.sample(
                    self.class_indices[i], (num_repl_batches+1)*class_num_batch_points[i]-last_batch_size
                )

                if last_batch_size == class_num_batch_points[i]:
                    class_batch_indices[i] = \
                        class_batch_indices[i] + chunk(repl_class_batch_indices, class_num_batch_points[i])
                elif last_batch_size < class_num_batch_points[i]:
                    repl_chunks = chunk(repl_class_batch_indices, class_num_batch_points[i])

                    class_batch_indices[i] = \
                        class_batch_indices[i][0:-1] + \
                        (torch.cat([class_batch_indices[i][-1], repl_chunks[-1]]),) + \
                        repl_chunks[0:-1]

        batches = []

        for j in range(self.num_batches):
            batch = []
            for i in range(self.num_classes):
                batch.extend(class_batch_indices[i][j].tolist())
            batches.append(batch)

        for batch in batches:
            random.shuffle(batch)

        return iter(batches)

    def __len__(self):
        return self.num_batches
