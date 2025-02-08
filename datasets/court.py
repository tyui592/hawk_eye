"""Tennis Court Data Code."""

import torch
import torchvision.transforms as T
import cv2
import numpy as np
import albumentations as A
from utils.misc import read_json_file
from utils.heatmap_ops import draw_gaussian


def get_transform(policy, imsize):
    """Get transform for court datasets."""
    height, width = imsize
    kp_params = A.KeypointParams(format='xy',label_fields=['class_labels'])
    if policy == 0:
        transform = A.Compose([
            A.Resize(width=width, height=height, p=1.0),
        ], keypoint_params=kp_params)

    elif policy == 1:
        transform = A.Compose([
            A.Perspective(scale=(0.05, 0.10), p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.Resize(width=width, height=height, p=1.0),
        ], keypoint_params=kp_params)
    return transform


class TennisCourtTransform:
    """Transform Class."""

    def __init__(self,
                 policy: int = 0,
                 imsize: list[int] = [576, 1024],
                 out_ch: int = 14,
                 stride: int = 4,
                 radius: int = 10):
        """Init."""

        self.transform = get_transform(policy, imsize)
        self.out_ch = out_ch
        self.radius = radius
        self.stride = stride

        self.normalize = T.Compose([
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __call__(self, img, kps, labels):
        """Transform the data."""
        # img = np.asarray(img)

        item = self.transform(image=img, keypoints=kps, class_labels=labels)
        img_h, img_w = item['image'].shape[:2]
        item['size'] = (img_h, img_w)

        # make a heatmap
        hm_h, hm_w = img_h // self.stride, img_w // self.stride
        heatmap = np.zeros((self.out_ch, hm_h, hm_w), dtype=np.float32)
        mask = np.zeros((self.out_ch, hm_h, hm_w), dtype=np.float32)
        for keypoint, label in zip(item['keypoints'], item['class_labels']):
            i = int(label)
            x, y = keypoint
            hx, hy = int(x // self.stride), int(y // self.stride)
            draw_gaussian(heatmap[i], (hx, hy), self.radius)
            
            mask[i, hy, hx] = 1.0
            
        item['heatmap'] = heatmap
        item['mask'] = mask
        item['image'] = self.normalize(item['image'])

        return item

class TennisCourtDataset:
    """Tennis Court Dataset."""

    def __init__(self, data_path, image_set, transform):
        """Init.

        image_set: 'train'/'val'
        """
        self.image_root = data_path / 'images'

        label_path = data_path / f'data_{image_set}.json'
        self.labels = read_json_file(label_path)

        self.transform = transform

    def __len__(self):
        """Len."""
        return len(self.labels)

    def __getitem__(self, idx):
        """Get a item."""
        data = self.labels[idx]
        file_id = data['id']
        image_path = self.image_root / f"{file_id}.png"
        img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        kps, labels = [], []
        for i, point in enumerate(data['kps']):
            x, y = point
            if 0 <= x < w and 0 <= y < h:
                kps.append(point)
                labels.append(str(i))
        # kps = data['kps']
        # labels = [str(i) for i in range(len(kps))]
        
        item = self.transform(img, kps, labels)
        item['raw_keypoints'] = kps
        item['raw_labels'] = labels
        item['file_id'] = file_id
        item['raw_size'] = img.shape[:2]
        item['raw_img'] = img
        return item


def collate_fn(batch):
    """Collate_fn for batch."""
    image_lst = [item['image'] for item in batch]
    mask_lst = [item['mask'] for item in batch]
    heatmap_lst = [item['heatmap'] for item in batch]
    keypoint_lst = [item['keypoints'] for item in batch]
    label_lst = [item['class_labels'] for item in batch]
    size_lst = [item['size'] for item in batch]
    raw_size_lst = [item['raw_size'] for item in batch]
    file_id_lst = [item['file_id'] for item in batch]
    raw_kps_lst = [item['raw_keypoints'] for item in batch]
    raw_labels_lst = [item['raw_labels'] for item in batch]
    raw_img_lst = [item['raw_img'] for item in batch]

    images = torch.stack(image_lst, dim=0)
    mask = torch.tensor(np.stack(mask_lst, axis=0))
    heatmaps = torch.tensor(np.stack(heatmap_lst, axis=0))
    inputs = {
        'image': images,
        'file_id': file_id_lst,
        'size': size_lst,
        'raw_size': raw_size_lst,
        'raw_img': raw_img_lst,
    }
    
    targets = {
        'heatmap': heatmaps,
        'class_label': label_lst,
        'keypoints': keypoint_lst,
        'mask': mask,
        'raw_keypoints': raw_kps_lst,
        'raw_labels': raw_labels_lst
    }
    return inputs, targets


def get_court_dataloader(args):
    """Get court dataloader for train/eval."""
    transform = TennisCourtTransform(policy=args.aug_policy,
                                     imsize=args.imsize,
                                     out_ch=args.num_class,
                                     stride=args.stride,
                                     radius=args.gaussian_radius)
    
    dataset = TennisCourtDataset(data_path=args.data_path,
                                 image_set=args.image_set,
                                 transform=transform)
    
    dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                             batch_size=args.batch_size,
                                             shuffle=args.mode == 'train',
                                             drop_last=args.mode == 'train',
                                             collate_fn=collate_fn)
    return dataloader