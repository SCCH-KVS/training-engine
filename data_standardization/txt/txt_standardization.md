# TXT format standardization

## General information
Rules on dataset description on TXT format.
This representation supports 2 task scenarios:
- Classification
- Segmentation
- Detection
- *In development: Traking, Multi-task*

This standardization describes following input cases:
- IC1: single TXT file

## Data loading to ALOHA toolflow
- IC1 case: please, use parameter `data_file` in the configuration and pick TXT file, which describes your dataset.

Please, be aware that all source locations in the TXT format standardization should contain ABSOLUTE path.

## Detailed standardization description
The standardization should be described as a list of paticular fileds separated by space depending on the task.

### Classification
File structure: `<image_file> <label>`
- `image_file` - absolute path to the image
- `label` - label description, integer or one-hot encoded
  >for integer description the count start from 0

### Segmentaiton
File structure: `<image_file> <mask_file>`
- `image_file` - absolute path to the image
- `mask_file` -  absolute path to the corresponding mask
  >segmentation areas should be setted to the integer values starting from 1 (0 - backgound) for a required number of classes in dataset

### Detection
File structure: `<image_file> <bounding_box>`
- `image_file` - absolute path to the image
- `bounding_box` - object boundaries
  >should be described as [Xmin, Ymin, Xmax, Ymax], where
  (Xmin, Ymin) pair specifying the lower-left corner<br>
  (Xmax, Ymax) pair specifying the upper-right corner<br>
  for more than one object should be represented as list if lists
