"""从官方入口获取原笔记中引用的公开示例数据。"""
import argparse
from pathlib import Path
from torchvision.datasets import CIFAR10
from torchvision.datasets.utils import download_and_extract_archive

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',choices=['cifar10','hymenoptera'],required=True)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]/'data'
    if args.dataset=='cifar10':
        for train in (True,False):
            dataset=CIFAR10(root/'cifar10',train=train,download=True)
            print('train' if train else 'test',len(dataset))
    else:
        download_and_extract_archive('https://download.pytorch.org/tutorial/hymenoptera_data.zip',str(root))
        for split in ('train','val'):
            for label in ('ants','bees'):
                p=root/'hymenoptera_data'/split/label
                print(p,len([x for x in p.iterdir() if x.suffix.lower() in {'.jpg','.jpeg','.png'}]))

if __name__=='__main__': main()
