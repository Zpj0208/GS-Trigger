# Dataset preparation

Raw image datasets are kept outside this package. The CIFAR readers use the official torchvision train/test splits. GTSRB uses the archived **26,640-image online-competition training subset**, with **12,630 final-test images and 43 classes**. A full final-training download would change the experiment.

## GTSRB

`gtsrb/labels/` retains 43 training label CSVs, the labeled and unlabeled final-test CSVs, and the original training Readme. The Readme identifies the online-competition subset. `training_index.csv` and `test_index.csv` give every expected path, label, image size, and archived server SHA-256. `original_server_sha256.txt` preserves the supplied server inventory.

1. Obtain the training-subset images and final-test images from the benchmark archive. Candidate links and their source are listed in `gtsrb/dataset.json`. These links were traced to the official torchvision reader; this package build did not download or verify those archives.
2. Arrange the selected images and supplied labels as follows. Preserve the supplied CSVs rather than replacing them with labels for a different split.

```text
<data-root>/gtsrb/
  Train/
    Readme.txt
    00000/
      GT-00000.csv
      00000_00000.ppm
      ...
    ...
    00042/
      GT-00042.csv
      ...
  Test/
    GT-final_test.csv
    GT-final_test.test.csv
    00000.ppm
    ...
```

3. Verify the image files against the archived subset:

```text
python scripts/verify_gtsrb.py --gtsrb-root <data-root>/gtsrb
```

Without `--gtsrb-root`, the same script checks the packaged annotations, labels, counts, and archived hash mapping. It reports zero images checked in that mode. With the option, it compares all 39,270 image files with the archived SHA-256 values.

The 26,640 training and 12,630 test indices must both pass. Do not use `GTSRB_Final_Training_Images.zip` as a silent substitute. The model's dataset factory expects `--dataset_path <data-root>` and appends `gtsrb`.

## CIFAR-10 and CIFAR-100

Use the original Python-format datasets from the [CIFAR dataset page](https://www.cs.toronto.edu/~kriz/cifar.html), or the torchvision download path used by the packaged reader. Both datasets have 50,000 training and 10,000 test inputs; CIFAR-10 has 10 classes and CIFAR-100 has 100. The factory appends `cifar10` or `cifar100` to `--dataset_path`. Keep those dataset directories under the parent data root.

Dataset images and annotations retain their source terms. This package does not assign them a new blanket software license.
