# data_standardization

Description of custom dataset representation accepted by ALOHA tool

Current support of data input into ALOHA tool is following:
- by picking `dataset`
    >allowed entries: MNIST, CIFAR10, CIFAR100, <custom>

    if `<custom>` is picked, user have two options:
    - from `data_file`

        > supported file formats: JSON, TXT

    - from `data_folder`

        > multiple file loading for classification and segmentation case

For more detailed description check corresponding folder

##### Important. Description provided in standardization files are not yet final. Changes my apply during development phase.
