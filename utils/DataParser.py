#!/usr/bin/env python
# -----------------------------------------------------------------------------
# Copyright (C) Software Competence Center Hagenberg GmbH (SCCH)
# All rights reserved.
# -----------------------------------------------------------------------------
# This document contains proprietary information belonging to SCCH.
# Passing on and copying of this document, use and communication of its
# contents is not permitted without prior written authorization.
# -----------------------------------------------------------------------------
# Created on : 2/20/2019 1:10 PM $ 
# by : shepeleva $ 
# SVN  $
#

# --- imports -----------------------------------------------------------------
import gc
import ast
import json
import h5py
import multiprocessing as mp
from functools import partial
from psutil import virtual_memory
from sklearn.model_selection import train_test_split
import time
from utils.utils import *


class DataParser:

    def __init__(self, data_set, data_file, data_folder, num_classes, img_size, normalize, zero_center, data_split, cross_val,
                 task_type, training_mode, multi_task, experiment_path, framework):
        self.data_set = data_set
        self.data_file = data_file
        self.data_folder = data_folder
        self.num_classes = num_classes
        self.img_size = img_size
        self.normalize = normalize
        self.zero_center = zero_center
        self.data_split = data_split
        self.cross_val = cross_val
        self.task_key = task_type
        self.is_training = training_mode
        self.multi_task = multi_task
        self.timestamp = str(time.time())
        self.experiment_path = experiment_path
        self.framework = framework

        if self.is_training:
            self._check_data_split()

        self._parse_data()

    def get_data_size(self):
        return self.data_size

    def get_file_name(self):
        return self.dict_data_path

    def get_log_name(self):
        return self.log_info_path

    def get_inference_data(self):
        return self.X_data, self.y_data

    def get_timestamp(self):
        return self.timestamp

    def get_ckpnt_path(self):
        return self.ckpnt_path

    def get_tr_path(self):
        return self.tr_path

    def _path_preparation(self):
        if self.experiment_path is None:
            self.experiment_path = os.path.join(os.path.dirname(os.path.abspath('utils')), "experiments")
        if not os.path.exists(self.experiment_path):
            os.makedirs(self.experiment_path)
        self.data_path = os.path.join(self.experiment_path, "datasets", self.data_set)
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)
        self.info_path = os.path.join(self.experiment_path, "info_logs", self.data_set)

        if self.is_training:
            if not os.path.exists(self.info_path):
                os.makedirs(self.info_path)
            self.tr_path = os.path.join(self.experiment_path, "train_logs", self.data_set, self.timestamp)
            if not os.path.exists(self.tr_path):
                os.makedirs(self.tr_path)
            self.ckpnt_path = os.path.join(self.experiment_path, "ckpnt_logs", self.data_set, self.timestamp)
            if not os.path.exists(self.ckpnt_path):
                os.makedirs(self.ckpnt_path)

    def _parse_data(self):

        file_name = self.data_set
        if self.img_size:
            file_name += '_resized({:d}x{:d})'.format(self.img_size[0], self.img_size[1])
        if self.normalize:
            file_name += '_norm'
        if self.zero_center:
            file_name += '_cent'
        file_name += '.hdf5'

        self._path_preparation()

        if self.is_training:
            log_file_name = os.path.join(self.info_path, self.timestamp + ".json")
        else:
            log_file_name = ''
        h5py_file_name = os.path.join(self.data_path, file_name)

        if self.data_set is "MNIST":
            self._load_mnist(h5py_file_name, log_file_name)
        elif self.data_set is "CIFAR10":
            self._load_cifar10(h5py_file_name, log_file_name)
        elif self.data_set is "CIFAR100":
            self._load_cifar100(h5py_file_name, log_file_name)
        elif self.data_set in ["", " ", None]:
            raise ValueError('Dataset name should be defined')
        else:
            if self.data_file:
                self._data_file_parse(h5py_file_name, log_file_name)
            elif self.data_folder:
                self._data_folder_parse(h5py_file_name, log_file_name)
            else:
                raise ValueError('No data presented')

    def _data_file_parse(self, h5py_file_name, log_file_name):
        if os.path.splitext(self.data_file)[1] == '.txt':
            x_list, y_list = self._process_txt()
        elif os.path.splitext(self.data_file)[1] == '.json':
            x_list, y_list = self._process_json()
        else:
            raise ValueError('Data format is not supported. Check documentation for data type support.')
        if self.is_training:
            if not os.path.isfile(h5py_file_name):
                # TODO: add normalization and zero-centering
                try:
                    self.slice_split = self._cross_val_split(len(x_list))
                    self.data_size = len(x_list)
                    self._dump_h5py(h5py_file_name, x_list, y_list)
                    self._dump_json_logs(h5py_file_name, log_file_name)
                    self.dict_data_path = h5py_file_name
                    self.log_info_path = log_file_name
                except KeyError:
                    raise ValueError("Unable to dump file")
            else:
                h5_file = h5py.File(h5py_file_name, 'r')
                self.data_size = len(h5_file['X_data'])
                self.slice_split = self._cross_val_split(len(h5_file['y_data']))
                h5_file.close()
                self._dump_json_logs(h5py_file_name, log_file_name)
                self.dict_data_path = h5py_file_name
                self.log_info_path = log_file_name
        else:
            self.X_data = x_list
            self.y_data = y_list

    def _process_txt(self):
        with open(self.data_file, 'r') as f:
            file_lines = f.readlines()
        line = file_lines[0].rstrip()
        if len(line.split()) == 1:
            if self.is_training:
                raise ValueError("not sufficient data for given task")
            else:
                if (os.path.splitext(line)[1].lower() in ['.jpg', '.jpeg', '.png', '.bmp']):
                    X_data_ = [line.rstrip().split()[0] for line in file_lines]
                    y_data_ = []
        else:
            if self.task_key is "classification":
                if (os.path.splitext(line.split()[0])[1].lower() in ['.jpg', '.jpeg', '.png', '.bmp']) and line.rstrip().split()[1].isdigit():
                    print("Classificaiton data detected. Integer encoded")
                    X_data_ = [line.rstrip().split()[0] for line in file_lines]
                    y_data_ = [int(line.rstrip().split()[1]) for line in file_lines]
                elif (os.path.splitext(line.split()[0])[1].lower() in ['.jpg', '.jpeg', '.png', '.bmp']) and isinstance(ast.literal_eval(''.join(line.rstrip().split()[1:])), list) and (sum(ast.literal_eval(''.join(line.rstrip().split()[1:]))) == 1):
                    print("Classificaiton data detected. One hot encoded")
                    X_data_ = [line.rstrip().split()[0] for line in file_lines]
                    y_data_ = [ast.literal_eval(''.join(line.rstrip().split()[1:])) for line in file_lines]
                else:
                    raise ValueError("Incorrect data representation")
            elif (self.task_key is "segmentation") or (self.task_key is "gan"):
                if (os.path.splitext(line.split()[0])[1].lower() in ['.jpg', '.jpeg', '.png', '.bmp']) and (
                        os.path.splitext(line.rstrip().split()[1])[1].lower() in ['.jpg', '.jpeg', '.png', '.bmp']):
                    X_data_ = [line.rstrip().split()[0] for line in file_lines]
                    y_data_ = [line.rstrip().split()[1] for line in file_lines]
                else:
                    raise ValueError("Incorrect data representation")
            elif self.task_key is "detection":
                if (os.path.splitext(line.split()[0])[1].lower() in ['.jpg', '.jpeg', '.png', '.bmp']) and isinstance(ast.literal_eval(''.join(line.rstrip().split()[1:])), list):
                    X_data_ = [line.rstrip().split()[0] for line in file_lines]
                    y_data_ = [ast.literal_eval(''.join(line.rstrip().split()[1:])) for line in file_lines]
                else:
                    raise ValueError("Incorrect data representation")
            else:
                raise ValueError('Such task not supported')

        return X_data_, y_data_

    def _process_json(self):
        with open(self.data_file, 'r') as f:
            file_lines = json.load(f)

        if file_lines['source_path'] in [None, 'null', '']:
            raise ValueError("Source path is not specified")
        else:
            if file_lines['mask_path'] in [None, 'null', '']:
                if self.task_key == "classification":
                    X_data_ = [os.path.join(file_lines['source_path'], d['frame']) for d in file_lines['meta']]
                    y_data_ = [o['object_class'] for d in file_lines['meta'] for o in d['objects']]
                elif self.task_key == "tracking":
                    X_data_ = [os.path.join(file_lines['source_path'], d['frame']) for d in file_lines['meta']]
                    y_data_ = [o['bb'] for d in file_lines['meta'] for o in d['objects']]
                elif self.task_key == "detection":
                    X_data_ = [os.path.join(file_lines['source_path'], d['frame']) for d in file_lines['meta']]
                    y_list = []
                    for d in file_lines['meta']:
                        lst = []
                        for o in d['objects']:
                            lst.append(o['bb'] + one_to_onehot(o['object_class'], self.num_classes))
                        y_list.append(lst)

                    ll = max([len(l) for l in y_list])
                    y_data_ = []
                    for l in y_list:
                        m = l
                        for i in range(ll - len(m)):
                            m.append([0, 0, 0, 0] + [0 for k in range(10)])
                        y_data_.append(m)
                    y_data_ = np.array(y_data_)
                else:
                    raise ValueError('Such task not supported')
            else:
                if (self.task_key == "segmentation") or (self.task_key == "gan"):
                    X_data_ = [os.path.join(file_lines['source_path'], d['frame']) for d in file_lines['meta']]
                    y_data_ = [os.path.join(file_lines['mask_path'], d['mask']) for d in file_lines['meta']]
                else:
                    raise ValueError('Such task not supported')

        return X_data_, y_data_

    def _data_folder_parse(self, h5py_file_name, log_file_name):
        path_list = [i for i in os.listdir(self.data_folder) if os.path.isdir(os.path.join(self.data_folder, i))]
        if len(path_list) < 2:
            if self.is_training:
                raise ValueError("not sufficient data for given task")
            else:
                x_list = path_walk(os.path.join(self.data_folder, path_list[0]))
                y_list = []
            # raise ValueError('Not sufficient number of folders')
        else:
            if self.task_key is "classification":
                x_list = []
                y_list = []
                for i in range(len(path_list)):
                    for dirpath, _, filenames in os.walk(path_list[i]):
                        for f in filenames:
                            if f.lower().endswith(tuple(['.jpg', '.jpeg', '.png', '.bmp'])):
                                x_list.append(os.path.abspath(os.path.join(dirpath, f)))
                                y_list.append(i)
                            else:
                                raise ValueError("Not supported data format")
            elif self.task_key is "segmentation":
                if "images" in path_list and "masks" in path_list:
                    x_list = path_walk(os.path.join(self.data_folder, 'images'))
                    y_list = path_walk(os.path.join(self.data_folder, 'masks'))
                else:
                    raise ValueError('Incorrect folder names for segmentation task')
            elif self.task_key is "gan":
                if "A" in path_list and "B" in path_list:
                    x_list = path_walk(os.path.join(self.data_folder, 'A'))
                    y_list = path_walk(os.path.join(self.data_folder, 'B'))
                else:
                    raise ValueError('Incorrect folder names for GAN task')
            else:
                raise ValueError('Such task not supported')

            # TODO: add normalization and zero-centering
        if self.is_training:
            if not os.path.isfile(h5py_file_name):
                try:
                    self.slice_split = self._cross_val_split(len(x_list))
                    self.data_size = len(x_list)
                    self._dump_h5py(h5py_file_name, x_list, y_list)
                    self._dump_json_logs(h5py_file_name, log_file_name)
                    self.dict_data_path = h5py_file_name
                    self.log_info_path = log_file_name
                except KeyError:
                    raise ValueError("Unable to dump file")
            else:
                h5_file = h5py.File(h5py_file_name, 'r')
                self.data_size = len(h5_file['X_data'])
                self.slice_split = self._cross_val_split(len(h5_file['y_data']))
                h5_file.close()
                self._dump_json_logs(h5py_file_name, log_file_name)
                self.dict_data_path = h5py_file_name
                self.log_info_path = log_file_name
        else:
            self.X_data = x_list
            self.y_data = y_list

    def _load_mnist(self, h5py_file_name, log_file_name):

        print('Creating h5py file for MNIST')
        if self.framework == 'tensorflow':
            import tensorflow as tf
            mnist = tf.keras.datasets.mnist
            if self.is_training:
                (X_data, y_data), _ = mnist.load_data()
            else:
                _, (self.X_data, self.y_data) = mnist.load_data()
            # y_data = one_hot_encode(y_data)
        elif self.framework == 'pytorch':
            import torchvision.datasets as datasets
            import struct
            if self.is_training:
                mnist_trainset = datasets.MNIST(root=os.path.join(self.data_path), train=True,
                                                download=True, transform=None)

                with open(os.path.join(self.data_path, self.data_set + r'/raw/train-labels-idx1-ubyte'), 'rb') as lbpath:
                    magic, n = struct.unpack('>II', lbpath.read(8))
                    y_data = np.fromfile(lbpath, dtype=np.uint8).tolist()

                with open(os.path.join(self.data_path, self.data_set + r'/raw/train-images-idx3-ubyte'), 'rb') as imgpath:
                    magic, num, rows, cols = struct.unpack(">IIII", imgpath.read(16))
                    X_data = np.fromfile(imgpath, dtype=np.uint8).reshape((-1, 28, 28, 1))
            else:
                mnist_trainset = datasets.MNIST(root=os.path.join(self.data_path, self.data_set), train=False, download=True,
                                                transform=None)
                with open(os.path.join(self.data_path, self.data_set + r'/raw/t10k-labels-idx1-ubyte'),
                          'rb') as lbpath:
                    magic, n = struct.unpack('>II', lbpath.read(8))
                    self.y_data = np.fromfile(lbpath, dtype=np.uint8).tolist()

                with open(os.path.join(self.data_path, self.data_set + r'/raw/t10k-images-idx3-ubyte'),
                          'rb') as imgpath:
                    magic, num, rows, cols = struct.unpack(">IIII", imgpath.read(16))
                    self.X_data = np.fromfile(imgpath, dtype=np.uint8).reshape((-1, 28, 28, 1))

        else:
            raise ValueError('Framework does not exist')

        if self.is_training:
            if not os.path.isfile(h5py_file_name):
                # TODO: add normalization and zero-centering
                try:
                    self.data_size = len(X_data)
                    self.slice_split = self._cross_val_split(len(y_data))
                    with h5py.File(h5py_file_name, 'a') as f:
                        f.create_dataset('X_data', data=X_data)
                        f.create_dataset('y_data', data=one_hot_encode(y_data))
                    self._dump_json_logs(h5py_file_name, log_file_name)
                    self.dict_data_path = h5py_file_name
                    self.log_info_path = log_file_name
                except KeyError:
                    raise ValueError("Unable to dump file")
            else:
                h5_file = h5py.File(h5py_file_name, 'r')
                self.data_size = len(h5_file['X_data'])
                self.slice_split = self._cross_val_split(len(h5_file['y_data']))
                h5_file.close()
                self._dump_json_logs(h5py_file_name, log_file_name)
                self.dict_data_path = h5py_file_name
                self.log_info_path = log_file_name

    def _load_cifar10(self, h5py_file_name, log_file_name):
        print('Creating h5py file for CIFAR10')
        if self.framework is 'tensorflow':
            import tensorflow as tf
            cifar10 = tf.keras.datasets.cifar10
            if self.is_training:
                (X_data, y_data), _ = cifar10.load_data()
            else:
                _, (X_data, y_data) = cifar10.load_data()

        elif self.framework is "pytorch":
            import torchvision.datasets as datasets
            if self.is_training:
                cifar_trainset = datasets.CIFAR10(root=os.path.join(self.data_path, self.data_set), train=True, download=True, transform=None)
                # works ONLY for Windows. I have NO idea why
                # X_data = cifar_trainset.train_data
                # y_data = cifar_trainset.train_labels
                X_data = [np.asarray(x) for x, _ in cifar_trainset]
                y_data = [x for _, x in cifar_trainset]
                [np.asarray(x) for x, _ in cifar_trainset]

            else:
                cifar_trainset = datasets.CIFAR10(root=os.path.join(self.data_path, self.data_set), train=False, download=True, transform=None)
                # self.X_data = cifar_trainset.test_data
                # self.y_data = cifar_trainset.test_labels
                self.X_data = [np.asarray(x) for x, _ in cifar_trainset]
                self.y_data = [x for _, x in cifar_trainset]

        else:
            raise ValueError('Framework does not exist')

        if self.is_training:
            if not os.path.isfile(h5py_file_name):
            # TODO: add normalization and zero-centering
                try:
                    self.data_size = len(X_data)
                    self.slice_split = self._cross_val_split(len(y_data))
                    with h5py.File(h5py_file_name, 'a') as f:
                        f.create_dataset('X_data', data=X_data)
                        f.create_dataset('y_data', data=one_hot_encode(y_data))
                    self._dump_json_logs(h5py_file_name, log_file_name)
                    self.dict_data_path = h5py_file_name
                    self.log_info_path = log_file_name
                except KeyError:
                    raise ValueError("Unable to dump file")
            else:
                h5_file = h5py.File(h5py_file_name, 'r')
                self.data_size = len(h5_file['X_data'])
                self.slice_split = self._cross_val_split(len(h5_file['y_data']))
                h5_file.close()
                self._dump_json_logs(h5py_file_name, log_file_name)
                self.dict_data_path = h5py_file_name
                self.log_info_path = log_file_name

    def _load_cifar100(self, h5py_file_name, log_file_name):
        if not os.path.isfile(h5py_file_name):
            print('Creating h5py file for CIFAR100')
            if self.framework is 'tensorflow':
                import tensorflow as tf
                if self.is_training:
                    cifar100 = tf.keras.datasets.cifar100
                    (X_data, y_data), _ = cifar100.load_data()
                else:
                    cifar100 = tf.keras.datasets.cifar100
                    _, (X_data, y_data) = cifar100.load_data()

            elif self.framework is "pytorch":
                import torchvision.datasets as datasets
                if self.is_training:
                    cifar_trainset = datasets.CIFAR100(root=os.path.join(self.data_path, self.data_set), train=True, download=True, transform=None)
                    y_data = [x for _, x in cifar_trainset]
                    [np.asarray(x) for x, _ in cifar_trainset]
                    # X_data = cifar_trainset.train_data
                    # y_data = cifar_trainset.train_labels
                else:
                    cifar_trainset = datasets.CIFAR100(root=os.path.join(self.data_path, self.data_set), train=True, download=True, transform=None)
                    # self.X_data = cifar_trainset.test_data
                    # self.y_data = cifar_trainset.test_labels
                    self.X_data = [np.asarray(x) for x, _ in cifar_trainset]
                    self.y_data = [x for _, x in cifar_trainset]

            else:
                raise ValueError('Framework does not exist')
        if self.is_training:
            if not os.path.isfile(h5py_file_name):
            # TODO: add normalization and zero-centering
                try:
                    self.data_size = len(X_data)
                    self.slice_split = self._cross_val_split(len(y_data))
                    with h5py.File(h5py_file_name, 'a') as f:
                        f.create_dataset('X_data', data=X_data)
                        f.create_dataset('y_data', data=one_hot_encode(y_data))
                    self._dump_json_logs(h5py_file_name, log_file_name)
                    self.dict_data_path = h5py_file_name
                    self.log_info_path = log_file_name
                    print("Dataset preparation finished")
                except KeyError:
                    raise ValueError("Unable to dump file")
            else:
                h5_file = h5py.File(h5py_file_name, 'r')
                self.data_size = len(h5_file['X_data'])
                self.slice_split = self._cross_val_split(len(h5_file['y_data']))
                h5_file.close()
                self._dump_json_logs(h5py_file_name, log_file_name)
                self.dict_data_path = h5py_file_name
                self.log_info_path = log_file_name

    def _dump_json_logs(self, h5py_file_name, log_file_name):
        log_dict = {'general_log': {'framework': self.framework,
                                    'task': self.task_key
                                    },
                    'data_log': {'data_path': h5py_file_name,
                                 'num_classes': self.num_classes,
                                 'img_size': self.img_size,
                                 'cross_val_split': self.slice_split,
                                 'normalization': self.normalize,
                                 'zero_center': self.zero_center
                                 },
                    'hyper_in_log': {},
                    'hyper_out_log': {}}
        try:
            with open(log_file_name, 'w') as f:
                json.dump(log_dict, f)
        except KeyError:
            raise ValueError("Unable to save logs")

    def _dump_h5py(self, h5py_file_name, x_list, y_list):
        mem = virtual_memory()
        free_mem = int(mem.free * 0.7)
        if (self.task_key is 'segmentation') or (self.task_key is 'gan'):
            pool = mp.Pool(mp.cpu_count())
            data_mem_x = sum(pool.map(count_size, x_list))
            pool.close()
            pool.join()
            pool = mp.Pool(mp.cpu_count())
            data_mem_y = sum(pool.map(count_size, y_list))
            pool.close()
            pool.join()


            x_list_split = chunk_split(x_list, data_mem_x//free_mem + 1)
            y_list_split = chunk_split(y_list, data_mem_y//free_mem + 1)

            with h5py.File(h5py_file_name, 'a') as f:
                dset_x = f.create_dataset('X_data', (0, self.img_size[0], self.img_size[1], self.img_size[2]),
                                          maxshape=(None, self.img_size[0], self.img_size[1], self.img_size[2]),
                                          chunks=True)
                dset_y = f.create_dataset('y_data', (0, self.img_size[0], self.img_size[1], self.img_size[2]),
                                          maxshape=(None, self.img_size[0], self.img_size[1], self.img_size[2]),
                                          chunks=True)
                for i in range(len(x_list_split)):
                    temp_map = np.zeros([len(x_list_split), self.img_size[0], self.img_size[1], self.img_size[2]])
                    pool = mp.Pool(mp.cpu_count())
                    func = partial(bulk_process, self.img_size)
                    temp_map = np.expand_dims(np.array(pool.map(func, x_list_split[i])), -1)
                    dset_x.resize(dset_x.shape[0] + len(x_list_split[i]), axis=0)
                    dset_x[-len(x_list_split[i]):] = temp_map
                    pool.close()
                    pool.join()
                for i in range(len(y_list_split)):
                    temp_map = np.zeros([len(x_list_split), self.img_size[0], self.img_size[1], self.img_size[2]])
                    pool = mp.Pool(mp.cpu_count())
                    func = partial(bulk_process, self.img_size)
                    temp_map = np.expand_dims(np.array(pool.map(func, x_list_split[i])), -1)
                    dset_y.resize(dset_y.shape[0] + len(y_list_split[i]), axis=0)
                    dset_y[-len(y_list_split[i]):] = temp_map
                    pool.close()
                    pool.join()

        else:
            start_process = time.time()
            with h5py.File(h5py_file_name, 'a') as f:
                pool = mp.Pool(mp.cpu_count())
                start_time = time.time()
                data_mem_x = sum(pool.map(count_size, x_list))
                pool.close()
                pool.join()
                elapsed_time = time.time() - start_time
                print("Data size count finished in")
                print(time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
                x_list_split = chunk_split(x_list, data_mem_x // free_mem + 1)

                print("chunk split finished")
                dset_x = f.create_dataset('X_data', (0, self.img_size[0], self.img_size[1], self.img_size[2]),
                                          maxshape=(None, self.img_size[0], self.img_size[1], self.img_size[2]),
                                          chunks=True, compression='gzip')
                for i in range(len(x_list_split)):
                    temp_map = np.zeros([len(x_list_split), self.img_size[0], self.img_size[1], self.img_size[2]])
                    print("Start data process loop " + str(i))
                    start_time = time.time()
                    pool = mp.Pool(mp.cpu_count())
                    func = partial(bulk_process, self.img_size)
                    temp_map = np.expand_dims(np.array(pool.map(func, x_list_split[i])),-1)
                    dset_x.resize(dset_x.shape[0] + len(x_list_split[i]), axis=0)
                    dset_x[-len(x_list_split[i]):] = temp_map
                    pool.close()
                    pool.join()
                    elapsed_time = time.time() - start_time
                    print(time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

                f.create_dataset('y_data', data=y_list)
            end_process = time.time() - start_process
            print('full time process: ' + time.strftime("%H:%M:%S", time.gmtime(end_process)))
        gc.collect()

    def _check_data_split(self):
        """

        :return:
        """
        if self.data_split:
            if (self.data_split > 1) or (self.data_split < 0.1):
                raise ValueError('Incorrect data_split value')
        else:
            if not self.cross_val:
                self.data_split = 0.7

    def _cross_val_split(self, data_dim):
        import random
        ind = random.sample([i for i in range(data_dim)], data_dim)
        if self.cross_val > 2 and self.cross_val < 10:
            cross_val_patch_size = int(data_dim/self.cross_val)
            slices = [ind[i:i+cross_val_patch_size] for i in range(0, data_dim, cross_val_patch_size)]
        else:
            print("Cross-validation is disabled. Will use data split instead")
            if not self.data_split and self.data_split > 1 and self.data_split < 0.1:
                print("Data split set up to 0.3")
                self.data_split = 0.3
                ind_train, ind_val = train_test_split(ind, test_size=self.data_split)
            else:
                ind_train, ind_val = train_test_split(ind, test_size=self.data_split)
            slices = [ind_train, ind_val]

        return slices


    @staticmethod
    def _preprocess_data_scope(img, msk=None, lbl=None, img_size=None, normalize=False, zero_center=False):
        """
        Preprocess data: zero-center, normalize, convert to one-hot
        :param img:
        :param msk:
        :param data_mean:
        :param data_std:
        :param normalize:
        :return:
        """
        # resize all images to same size
        if img_size:
            img_shapes = [x.shape for x in img]
            # check if all images are equal to img_size
            if not np.all(np.array(img_shapes) == img_size):
                X_resized = []
                for x in img:
                    # resize data to given image size
                    if np.any(np.array(img_size) != x.shape):
                        X_resized.append(resize_images_gpu(x, img_size))
                img = X_resized
                if msk:
                    y_resized = []
                    for y_ in msk:
                        # resize data to given image size
                        if np.any(np.array(img_size) != y_.shape):
                            y_resized.append(resize_images_gpu(y_, img_size))
                    msk = y_resized
        else:
            img_shapes = [x.shape for x in img]
            if not np.all(np.array(img_shapes) == img_shapes[0]):
                raise ValueError(
                    "All images and masks should be same size! Add image_size to config to reshape all images")

        # check data range and convert to 0-1 range
        if zero_center:
            if np.max(img) > 1:
                print('Data range [{:.2f}, {:.2f}]'.format(np.min(img), np.max(img)))
                print('[PREPROCESS]\tConverting to [0, 1]')
                img = [x / 255 for x in img]
                assert (np.min(img) >= 0)
                assert (np.max(img) <= 1)
            # zero-center data
            data_mean = np.mean(img)
            img -= data_mean
            print('[PREPROCESS]\tZero-centered data with data mean {:.2f} to min: {:.2f}, max: {:.2f}'
                  .format(data_mean, img.min(), img.max()))

        if normalize:
            img /= np.std(img)
            print('[PREPROCESS]\tNormalizing data by dividing by train STD! (std: {:.2f} - min: {:.2f}, max: {:.2f}'
                  .format(np.std(img), img.min(), img.max()))
        # convert labels to one-hot if not already
        if lbl:
            if isinstance(lbl[0], int):
                lbl = one_hot_encode(lbl)
        return img, msk, lbl
