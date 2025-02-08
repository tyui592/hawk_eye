import torch
import torchvision.transforms as T
import cv2
import numpy as np
import albumentations as A
from collections import defaultdict
from utils.misc import load_pickle
from utils.heatmap_ops import draw_gaussian


class TennisBallDataset:
    def __init__(self, 
                 transform,
                 data_path,
                 image_set: str = 'train',
                 num_seq: int = 3,
                 include_lv2: bool = True):
        self.transform = transform
        self.data_path = data_path
        self.num_seq = num_seq
        self.foreground = ['1']
        if include_lv2:
            self.foreground.append('2')
                
        label_path = data_path / f'data_{image_set}_clip.pkl'
        labels = load_pickle(label_path)
        self.labels = labels
        self.data = []
        for clip in labels:
            seq_data = self.chunks(clip, self.num_seq)
            self.data += seq_data

    def __len__(self):
        return len(self.data)
            
    def __getitem__(self, idx):
        samples = self.data[idx]
        
        item = defaultdict(list)
        for sample in samples:
            img_path = self.data_path / sample['file name']
            img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            h, w = img.shape[:2]
            item['images'].append(img)
            visibility = sample['visibility']
            point = [-1, -1]
            if visibility in self.foreground:
                x = int(sample['x-coordinate'])
                y = int(sample['y-coordinate'])
                if 0 <= x < w and 0 <= y < h:                    
                    point = [x, y]
            item['keypoints'].append(point)
            item['raw_size'].append(img.shape[:2])
            item['file_name'].append(sample['file name'])
            
        # data transformation
        transformed = self.transform(item)
        
        item['image'] = transformed['input']
        item['mask'] = transformed['mask']
        item['heatmap'] = transformed['heatmap']
        
        out = {
            'image': transformed['input'],
            'mask': transformed['mask'],
            'heatmap': transformed['heatmap'],
            'file_names': item['file_name'],
            'raw_size': item['raw_size'][self.num_seq-1],
            'raw_keypoints': item['keypoints'][self.num_seq-1],
            'size': transformed['size'],
        }
            
        return out
    
    def chunks(self, lst, n, step=1):
        """
        주어진 리스트에서 n개의 연속된 원소를 step 간격으로 이동하며 새로운 chunk 리스트를 생성
        """
        if n <= 0 or n > len(lst):
            raise ValueError("n은 1 이상이고, input_list 길이 이하이어야 합니다.")
        if step <= 0:
            raise ValueError("step은 1 이상이어야 합니다.")

        # step 간격으로 n개의 연속된 원소를 슬라이싱
        chunks = [lst[i:i + n] for i in range(0, len(lst) - n + 1, step)]
        return chunks
    
def get_transform(policy, imsize):
    """Get transform for court datasets."""
    height, width = imsize
    kp_params = A.KeypointParams(format='xy', remove_invisible=False)
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

class TennisBallTransform:
    """Transform Class."""

    def __init__(self,
                 policy: int = 0,
                 imsize: list[int] = [576, 1024],
                 out_ch: int = 1,
                 stride: int = 4,
                 radius: int = 10,
                 num_seq: int = 3):
        """Init."""
        self.transform = get_transform(policy, imsize)
        self.out_ch = out_ch
        self.radius = radius
        self.stride = stride
        self.num_seq = num_seq
        self.normalize = T.Compose([
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
    def __call__(self, item):
        # print('Raw', item['keypoints'])
        transformed = self.transform(images=item['images'], keypoints=item['keypoints'])
        img_h, img_w = transformed['images'][0].shape[:2]
        
        hm_h, hm_w = int(img_h // self.stride), int(img_w // self.stride)
        heatmap = np.zeros((self.out_ch, hm_h, hm_w), dtype=np.float32)
        mask = np.zeros((self.out_ch, hm_h, hm_w), dtype=np.float32)

        # make a heatmap with the last input image
        keypoint = transformed['keypoints']    
        x, y = keypoint[self.num_seq-1]
        if 0 <= x < img_w and 0 <= y < img_h:
            hx, hy = int(x // self.stride), int(y // self.stride)
            draw_gaussian(heatmap[0], (hx, hy), self.radius)
            mask[0, hy, hx] = 1.0
        
        # channel wise concatenation
        tensor = torch.cat([self.normalize(img) for img in transformed['images']], dim=0)
        transformed['heatmap'] = torch.tensor(heatmap)
        transformed['mask'] = torch.tensor(mask)
        transformed['input'] = tensor
        transformed['size'] = (img_h, img_w)
        return transformed
    
def collate_fn(batch):
    image_lst = [item['image'] for item in batch]
    mask_lst = [item['mask'] for item in batch]
    heatmap_lst = [item['heatmap'] for item in batch]
    size_lst = [item['size'] for item in batch]
    raw_size_lst = [item['raw_size'] for item in batch]
    file_names_lst = [item['file_names'] for item in batch]
    raw_kps_lst = [item['raw_keypoints'] for item in batch]
    inputs = {
        'image': torch.stack(image_lst, dim=0),
        'file_name': file_names_lst,
        'size': size_lst,
        'raw_size': raw_size_lst,
    }
    targets = {
        'heatmap': torch.stack(heatmap_lst, dim=0),
        'mask': torch.stack(mask_lst, dim=0),
        'raw_keypoints': raw_kps_lst
    }
    return inputs, targets


def get_ballseq_dataloader(args):
    transform = TennisBallTransform(policy=args.aug_policy,
                                    imsize=args.imsize,
                                    out_ch=args.num_class,
                                    stride=args.stride,
                                    num_seq=args.num_seq,
                                    radius=args.gaussian_radius)
    
    dataset = TennisBallDataset(data_path=args.data_path,
                                image_set=args.image_set,
                                transform=transform,
                                num_seq=args.num_seq)
                                
    dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                             batch_size=args.batch_size,
                                             shuffle=args.mode == 'train',
                                             drop_last=args.mode == 'train',
                                             collate_fn=collate_fn)
    return dataloader