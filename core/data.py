import random
import torch
import torch.distributed as dist
from torchvision import datasets, transforms

class Partition(object):

    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]


class DataPartitioner(object):

    def __init__(self, data, sizes=[0.7, 0.2, 0.1], seed=7548):
        self.data = data
        self.partitions = []
        rng = random
        rng.seed(seed)
        data_len = len(data)
        indexes = [x for x in range(0, data_len)]
        rng.shuffle(indexes)

        for frac in sizes:
            part_len = int(frac * data_len)
            self.partitions.append(indexes[0:part_len])
            indexes = indexes[part_len:]

    def use(self, partition):
        return Partition(self.data, self.partitions[partition])


def get_dataset(type, train = True):
    dataset = None
    
    rank = 0
    if dist.is_initialized():
        rank = dist.get_rank()
    
    if type == "imagenet":
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if train:
            dataset = datasets.ImageFolder(f'/home/onycom/imagenet/ILSVRC_{ rank }/Data/CLS-LOC/train/', transform=transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ]))
        else:
            dataset = datasets.ImageFolder(f'/home/onycom/imagenet/ILSVRC_{ rank }/Data/CLS-LOC/val/', transform=transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.ToTensor(),
                normalize
            ]))


    elif type == "cifar100":
        normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        if train:
            dataset = datasets.CIFAR100(f'~/dataset/CIFAR100_{ rank }/', train=True, download=False,  transform=transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ]))
        else:
            dataset = datasets.CIFAR100(f'~/dataset/CIFAR100_{ rank }/', train=False, download=False, transform=transforms.Compose([
                #transforms.Resize(224),
                transforms.ToTensor(),
                normalize
            ]))
            
    # dataset = datasets.MNIST(f'~/dataset{ dist.get_rank() }/', train=True, download=True, transform=transforms.Compose([
    #    transforms.ToTensor(),
    #    transforms.Normalize((0.1307,), (0.3081, ))
    # ]))
    return dataset

def partition_dataloader(dataset, batch_size = 1024, seed=7548, num_workers = 6, shuffle = True):
    rank = 0
    size = 1
    if dist.is_initialized():
        rank = dist.get_rank()
        size = dist.get_world_size()

    bsz = batch_size / float(size)
    partition_sizes = [1.0 / size for _ in range(size)]
    partition = DataPartitioner(dataset, partition_sizes, seed)
    partition = partition.use(rank)
    loader = torch.utils.data.DataLoader(partition, batch_size = int(bsz), shuffle=shuffle,  num_workers=num_workers, pin_memory=True)
    
    return loader, bsz

# class ImageNetValDataset(Dataset): 
#     def __init__(self, data_path, transform=None):
#         self.transform = transform
#         self.data_path = data_path
        
#         y_list = []
#         self.targets = []
#         self.x_path_list = []
#         with open(data_path,'r') as data_list:
#             for row in data_list.readlines():
#                 filename = row.split(' ')[0]
#                 anno_file = os.path.join(os.path.dirname(data_path).replace('ImageSets', 'Annotations'),'val', filename) + '.xml'
#                 data_file = os.path.join(os.path.dirname(data_path).replace('ImageSets', 'Data'), 'val', filename) + '.JPEG'
#                 with open(anno_file, 'r') as anno_f:
#                     xml = anno_f.read()
#                     xml = xmltodict.parse(xml)
#                     if type(xml['annotation']['object']) is list:
#                         y = xml['annotation']['object'][0]['name'] # class
#                     else:
#                         y = xml['annotation']['object']['name'] # class
                        
#                     y_list.append(y)
#                 self.x_path_list.append(data_file)
        
#         self.classes = list(set(y_list))
#         self.classes.sort()
        
#         self.class_to_idx = {}
#         for idx, item in enumerate(self.classes):
#             self.class_to_idx[item] = idx
        
        
#         for item in y_list:
#             self.targets.append(self.class_to_idx[item])
        
#         self.data_count = len(self.class_to_idx)
        

#   # 총 데이터의 개수를 리턴
#     def __len__(self):
#         return self.data_count

#   # 인덱스를 입력받아 그에 맵핑되는 입출력 데이터를 파이토치의 Tensor 형태로 리턴
#     def __getitem__(self, idx): 
#         if torch.is_tensor(idx):
#             idx = idx.tolisT()
        
#         img = Image.open(self.x_path_list[idx])
#         if self.transform != None:
#             img = self.transform(img)
            
#         y = self.class_to_idx[idx]
#         return img, y
