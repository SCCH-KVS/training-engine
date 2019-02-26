# JSON format standardization

## General information
Rules on dataset description on JSON format.
This representation supports 3 task scenarios:
- Classification
- Segmentation
- Detection
- *In development: Traking, Multi-task*

This standardization describes following input cases:
- IC1: single folder with images
- IC2: multiple folders with images
- IC3: single video
- IC4: multiple videos

## Data loading to ALOHA toolflow
By using JSON format standardization, please, use parameter `data_file` in the configuration and pick JSON file, which describes your dataset.

Please, be aware that all source locations in the JSON format standardization should contain ABSOLUTE path.


## Detailed standardization description
The standardization should be described as a dictionary of `<key>: <value>` pairs.

Allowed list of `key` values:

- `name` - dataset name
    >str value, describes dataset name, Ex.: 'random_set'
- `width` - width of the frame/image
    >set value only if all frames/images of the same size, otherwise put 0
- `height` - height of the frame/image
    >set value only if all frames/images of the same size, otherwise put 0
- `source` - source of the information (`video` or `image`)
    >describes which source will be used, avaliable values:  `video` or `image`
- `source_path` - absolute path to the source
    >absolute or relative to the .json file path to the data source
- `mask_path` - absolute path to masks if presented
    >absolute or relative to the .json file path to the masks (only for Segmentation task)
- `data_size` - number of frames per video / number of images in dataset
    >depending on IC representaiton this value coud difer use following example: <br>
    for IC1 or IC3: 2340 - means that dataset represented by one folder with 2340 images in total OR by one video with 2340 frames in total<br>
    for IC2 or IC4: [345, 654, 123] - means that dataset represented by 3 folders with 345, 654 and 123 images accordingly OR by 3 videos with 345, 654 and 123 frames accordingly
- `meta`- detailed information for specific task
    >in this section every frame/image should get detailed description accordingly to the task

###  `meta` for Classification
List of following dictionaries:
- `frame` - frame/image id
    >for IC1: image_name<br>
    for IC2: [image_folder, image]<br>
    for IC3: frame_id<br>
    for IC4: [video_file_name, frame_id]
- `width`
    >width of particular frame, if global width is not setted up
- `height`
    >height of particular frame, if global height is not setted up
- `mask` - mask presence
    >for this task should be equal to null
- `objects` - list of the descriptions of objects which should be classified in current frame
    - `object_class` - object class indentification
      >integer or one-hot labeling
    - `bb` - bounding box
      >for this task should be equal to null
    - `polygon` - polygon area
      >for this task should be equal to null

###  `meta` for Segmentation
List of following dictionaries:
- `frame` - frame/image id
    >for IC1: image_name<br>
    for IC2: [image_folder, image]<br>
    for IC3: frame_id<br>
    for IC4: [video_file_name, frame_id]
- `width`
    >width of particular frame, if global width is not setted up
- `height`
    >height of particular frame, if global height is not setted up
- `mask`- mask presence, video/image representation might be independent from frame representaiton
    >for IC1: image_name<br>
    for IC2: [image_folder, image]<br>
    for IC3: frame_id<br>
    for IC4: [video_file_name, frame_id]
- `objects` - list of the descriptions of objects which should be classified in current frame
    - `object_class` - object class indentification
      >integer or one-hot labeling
    - `bb` - bounding box
      >for this task should be equal to null
    - `polygon` - polygon area
      >list of points describing polygon

### `meta` for Detection
List of following dictionaries:
- `frame` - frame/image id
    >for IC1: image_name<br>
    for IC2: [image_folder, image]<br>
    for IC3: frame_id<br>
    for IC4: [video_file_name, frame_id]
- `width`
    >width of particular frame, if global width is not setted up
- `height`
    >height of particular frame, if global height is not setted up
- `mask` - mask presence
    >for this task should be equal to null
- `objects` - list of the descriptions of objects which should be classified in current frame
    - `object_class` - object class indentification
      >integer or one-hot labeling
    - `bb` - object boundaries
      >should be described as [Xmin, Ymin, Xmax, Ymax], where<br>
      (Xmin, Ymin) pair specifying the lower-left corner<br>
      (Xmax, Ymax) pair specifying the upper-right corner
    - `polygon` - polygon area
      >list of points describing polygon
