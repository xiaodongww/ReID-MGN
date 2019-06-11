from torchvision import transforms
from torch.utils.data import dataset, dataloader
from torchvision.datasets.folder import default_loader
from utils.RandomErasing import RandomErasing
from utils.RandomSampler import RandomSampler
from opt import opt
import os
import re
import glob


class Data():
    def __init__(self, name='market1501'):
        print('loading {}'.format(name))
        train_transform = transforms.Compose([
            transforms.Resize((384, 128), interpolation=3),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            RandomErasing(probability=0.5, mean=[0.0, 0.0, 0.0])
        ])

        test_transform = transforms.Compose([
            transforms.Resize((384, 128), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.name = name
        if name == 'market1501':
            self.trainset = Market1501(train_transform, 'train', opt.data_path)
            self.testset = Market1501(test_transform, 'test', opt.data_path)
            self.queryset = Market1501(test_transform, 'query', opt.data_path)
        elif name == 'csm':
            self.trainset = CSM(train_transform, 'train', opt.data_path)
            self.testset = CSM(test_transform, 'test', opt.data_path)
            self.queryset = CSM(test_transform, 'query', opt.data_path)
        else:
            raise KeyError('Only support market1501 and csm, {} not supported'.format(name))
        self.train_loader = dataloader.DataLoader(self.trainset,
                                                  sampler=RandomSampler(self.trainset, batch_id=opt.batchid,
                                                                        batch_image=opt.batchimage),
                                                  batch_size=opt.batchid * opt.batchimage, num_workers=8,
                                                  pin_memory=True)
        self.test_loader = dataloader.DataLoader(self.testset, batch_size=opt.batchtest, num_workers=8, pin_memory=True)
        self.query_loader = dataloader.DataLoader(self.queryset, batch_size=opt.batchtest, num_workers=8,
                                                  pin_memory=True)
        if opt.mode == 'vis':
            self.query_image = test_transform(default_loader(opt.query_image))


class Market1501(dataset.Dataset):
    def __init__(self, transform, dtype, data_path):

        self.transform = transform
        self.loader = default_loader
        self.data_path = data_path

        if dtype == 'train':
            self.data_path += '/bounding_box_train'
        elif dtype == 'test':
            self.data_path += '/bounding_box_test'
        else:
            self.data_path += '/query'

        self.imgs = [path for path in self.list_pictures(self.data_path) if self.id(path) != -1]

        self._id2label = {_id: idx for idx, _id in enumerate(self.unique_ids)}

    def __getitem__(self, index):
        path = self.imgs[index]
        target = self._id2label[self.id(path)]

        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.imgs)

    @staticmethod
    def id(file_path):
        """
        :param file_path: unix style file path
        :return: person id
        """
        return int(file_path.split('/')[-1].split('_')[0])

    @staticmethod
    def camera(file_path):
        """
        :param file_path: unix style file path
        :return: camera id
        """
        return int(file_path.split('/')[-1].split('_')[1][1])

    @property
    def ids(self):
        """
        :return: person id list corresponding to dataset image paths
        """
        return [self.id(path) for path in self.imgs]

    @property
    def unique_ids(self):
        """
        :return: unique person ids in ascending order
        """
        return sorted(set(self.ids))

    @property
    def cameras(self):
        """
        :return: camera id list corresponding to dataset image paths
        """
        return [self.camera(path) for path in self.imgs]

    @staticmethod
    def list_pictures(directory, ext='jpg|jpeg|bmp|png|ppm|npy'):
        assert os.path.isdir(directory), 'dataset is not exists!{}'.format(directory)

        return sorted([os.path.join(root, f)
                       for root, _, files in os.walk(directory) for f in files
                       if re.match(r'([\w]+\.(?:' + ext + '))', f)])

class CSM(dataset.Dataset):
    def __init__(self, transform, dtype, data_path):

        self.transform = transform
        self.loader = default_loader
        self.data_path = data_path

        if dtype == 'train':
            self.data_path += '/train_bbox'
        elif dtype == 'test':
            self.data_path += '/val_bbox'
        else:
            self.data_path += '/val_bbox'

        print('loading {} set from {}'.format(dtype, self.data_path))

        if dtype == 'train':
            self.imgs = [path for path in self.list_pictures(self.data_path) if self.id(path) != -1]
        else:
            # The val set is too large, we select the first 10 imgs
            imgs = [path for path in self.list_pictures_of_query(self.data_path) if self.id(path) != -1]
            self.imgs = imgs[:10000]

        # self.imgs = [path for path in self.list_pictures(self.data_path) if self.id(path) != 'other']

        self._id2label = {_id: idx for idx, _id in enumerate(self.unique_ids)}

    def __getitem__(self, index):
        path = self.imgs[index]
        target = self._id2label[self.id(path)]

        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.imgs)

    @staticmethod
    def id(file_path):
        """
        :param file_path: unix style file path
        :return: person id
        """
        person_id_str, movie_id, _, _ = os.path.basename(file_path).split('_')
        if person_id_str[:5] == 'other':
            pid = -1
        else:
            pid = int(person_id_str[2:]) # person_id_str e.g.: nm1913734
        return pid
    # @staticmethod
    # def id(file_path):
    #     """
    #     :param file_path: unix style file path
    #     :return: person id
    #     """
    #     return
        # return int(file_path.split('/')[-1].split('_')[0])

    @staticmethod
    def camera(file_path):
        """
        :param file_path: unix style file path
        :return: camera id
        no camera for CSM dataset, so we just generate a unique id for each image
        """
        return id(file_path)
        # return int(file_path.split('/')[-1].split('_')[1][1])

    @property
    def ids(self):
        """
        :return: person id list corresponding to dataset image paths
        """
        return [self.id(path) for path in self.imgs]

    @property
    def unique_ids(self):
        """
        :return: unique person ids in ascending order
        """
        return sorted(set(self.ids))

    @property
    def cameras(self):
        """
        :return: camera id list corresponding to dataset image paths
        """
        return [self.camera(path) for path in self.imgs]

    @staticmethod
    def list_pictures(directory, ext='jpg|jpeg|bmp|png|ppm|npy'):
        assert os.path.isdir(directory), 'dataset is not exists!{}'.format(directory)

        return sorted([os.path.join(root, f)
                       for root, _, files in os.walk(directory) for f in files
                       if re.match(r'([\w]+\.(?:' + ext + '))', f)])

    @staticmethod
    def list_pictures_of_query(directory):
        img_paths = []
        for movie_dir in os.listdir(directory):
            image_dir = os.path.join(directory, '{}/candidates'.format(movie_dir))
            img_paths_part = glob.glob(os.path.join(image_dir, 'nm*.jpg'))
            img_paths.extend(img_paths_part)
        return img_paths


# class CSM(dataset.Dataset):
#     """
#     CMS dataset, used for WIDER FACE cast search challenge
#     Reference:
#
#     Dataset statistics:
#     # identities: # in total, we ignore images with 'other' identity
#     # images: * (train) + * (query) + * (gallery)
#     """
#
#     def __init__(self, dtype, root='/home/haoluo/data', verbose=True, transform=None):
#         super(CSM, self).__init__()
#         self.root = root
#         if dtype == 'train':
#             self.img_dir = os.path.join(root, 'train_bbox')
#         elif dtype == 'test':
#             self.img_dir = os.path.join(root, 'val_bbox')
#         else:
#             self.img_dir = os.path.join(root, 'val_bbox')
#         self.transform = transform
#
#         self.data = self._process_dir(self.img_dir, relabel=True, dtype=dtype)
#
#
#     def __getitem__(self, index):
#         img_path, target, movie_id = self.data[index]
#
#         img = self.loader(img_path)
#         if self.transform is not None:
#             img = self.transform(img)
#
#         return img, target
#
#     def __len__(self):
#         return len(self.imgs)
#
#
#     def _process_dir(self, dir_path, relabel=False, dtype='train'):
#         img_paths = []
#         for movie_dir in os.listdir(dir_path):
#             image_dir = os.path.join(dir_path, '{}/candidates'.format(movie_dir))
#             img_paths_part = glob.glob(os.path.join(image_dir, 'nm*.jpg'))
#             img_paths.extend(img_paths_part)
#
#         if dtype == 'test':
#             img_paths = img_paths[:min(len(img_paths), 10000)]  # use 10k images for evaluation,
#
#         pid_container = set()
#         dataset = []
#         for img_path in img_paths:
#             img_name = os.path.basename(img_path)
#             pid, movie_id, _, _ = img_name.split('_')
#             pid_container.add(pid)
#
#         pid2label = {pid: label for label, pid in enumerate(pid_container)}
#         for img_path in img_paths:
#             img_name = os.path.basename(img_path)
#             pid, movie_id, _, _ = img_name.split('_')
#             if relabel:
#                 pid = pid2label[pid]
#             dataset.append((img_path, pid, movie_id))
#         return dataset
#
#     @property
#     def cameras(self):
#         """
#         :return: camera id list corresponding to dataset image paths
#         for CSM dataset, there is no concept of camera, so we generate a unique id for each image
#         """
#         return [id(img_path) for img_path, target, movie_id in self.data]
#
#     #
#     # @staticmethod
#     # def id(file_path):
#     #     """
#     #     :param file_path: unix style file path
#     #     :return: person id
#     #     """
#     #     pid, movie_id, _, _ = os.path.basename(file_path).split('_')
#     #     return int(pid)
#     #
