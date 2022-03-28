3DMatch dataset metadata, obtained from Predator with slight modifications.
    https://github.com/overlappredator/OverlapPredator

- *.pkl: Contains point cloud pairs indices, same as the files provided in Predator. Note however that one point cloud (train/7-scenes-fire/cloud_bin_19.pth) in the training dataset has a wrong pose, so that is removed from train_info.pkl.
- benchmarks/
  - contain groundtruth poses (in same format as 3DMatch official test set) for evaluating 3DMatch and 3DLoMatch performance.