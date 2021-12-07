from dataloader import DuckietownDataset
import numpy as np
import torch

K = np.array([[373.2779426913342, 0.0, 318.29785021099894],
                  [0.0, 367.9439633567062, 263.9058079734077],
                  [0.0, 0.0, 1.0]])
dataset = DuckietownDataset("alex_2small_loops_ground_truth.txt", "alex_2small_loops_images", K=K)

test_dataloader = torch.utils.data.DataLoader(dataset, batch_size=5,
                                          shuffle=True, num_workers=0)


for data, rel_pos, K in test_dataloader:
    print(data.shape, rel_pos)
    break