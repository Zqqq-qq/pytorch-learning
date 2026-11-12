"""从仓库任意位置运行：真实 CIFAR-10 训练，或 --synthetic 离线流程检查。"""
import argparse
import json
from pathlib import Path
import random
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3,32,5,padding=2),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(32,32,5,padding=2),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(32,64,5,padding=2),nn.ReLU(),nn.MaxPool2d(2),
            nn.Flatten(),nn.Linear(1024,64),nn.ReLU(),nn.Linear(64,10))
    def forward(self,x): return self.layers(x)

def prepare_image():
    # 训练图已是 32×32；推理图先 resize，再执行相同的 ToTensor。
    return transforms.Compose([transforms.Resize((32,32)),transforms.ToTensor()])

def evaluate(model,loader,device):
    model.eval(); total_loss=0.; correct=0; count=0
    criterion=nn.CrossEntropyLoss(reduction='sum')
    with torch.no_grad():
        for x,y in loader:
            x,y=x.to(device),y.to(device)
            logits=model(x)
            total_loss+=criterion(logits,y).item()
            correct+=(logits.argmax(1)==y).sum().item(); count+=len(y)
    return {'loss':total_loss/count,'accuracy':correct/count,'samples':count}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epochs',type=int,default=5)
    parser.add_argument('--batch-size',type=int,default=64)
    parser.add_argument('--synthetic',action='store_true')
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--predict',type=Path)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    if args.epochs<1 or args.batch_size<1: parser.error('epochs 和 batch-size 必须为正数')
    random.seed(42); torch.manual_seed(42)
    # CPU 限制线程是为了让小规模演示的耗时稳定；GPU 路径仍自动选 CUDA。
    torch.set_num_threads(2)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=Classifier().to(device)
    checkpoint=args.checkpoint or ROOT/'checkpoints'/('synthetic.pt' if args.synthetic else 'cifar10.pt')
    if args.predict:
        saved=torch.load(checkpoint,map_location=device,weights_only=True)
        if saved['dataset']!='cifar10':
            raise ValueError('合成数据 checkpoint 不能用作真实图像分类，请先训练 CIFAR-10')
        model.load_state_dict(saved['model']); model.eval()
        with Image.open(args.predict) as image:
            x=prepare_image()(image.convert('RGB')).unsqueeze(0).to(device)
        with torch.no_grad(): logits=model(x)
        index=logits.argmax(1).item()
        print(json.dumps({'class_index':index,'class':saved['classes'][index]},ensure_ascii=False))
        return
    name='synthetic' if args.synthetic else 'cifar10'
    if args.synthetic:
        print('合成随机数据：只验证流程，不代表 CIFAR-10 成绩。')
        generator=torch.Generator().manual_seed(42)
        train=TensorDataset(torch.rand(80,3,32,32,generator=generator),torch.randint(10,(80,),generator=generator))
        test=TensorDataset(torch.rand(33,3,32,32,generator=generator),torch.randint(10,(33,),generator=generator))
    else:
        train=datasets.CIFAR10(ROOT/'data/cifar10',train=True,download=True,transform=prepare_image())
        test=datasets.CIFAR10(ROOT/'data/cifar10',train=False,download=True,transform=prepare_image())
    trainloader=DataLoader(train,batch_size=args.batch_size,shuffle=True,num_workers=0)
    testloader=DataLoader(test,batch_size=args.batch_size,shuffle=False,num_workers=0,drop_last=False)
    optimizer=torch.optim.SGD(model.parameters(),lr=0.01,momentum=0.9)
    start=0
    if args.resume:
        saved=torch.load(checkpoint,map_location=device,weights_only=True)
        if saved['dataset']!=name: raise ValueError('恢复训练的数据类型与 checkpoint 不匹配')
        model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
        start=saved['epoch']; torch.set_rng_state(saved['torch_rng_state'].cpu())
        random.setstate(saved['python_rng_state'])
        if device.type=='cuda' and saved.get('cuda_rng_state') is not None:
            torch.cuda.set_rng_state_all(saved['cuda_rng_state'])
    criterion=nn.CrossEntropyLoss()
    checkpoint.parent.mkdir(parents=True,exist_ok=True)
    writer=SummaryWriter(str(ROOT/'runs'/name),purge_step=start if args.resume else None)
    try:
        for epoch in range(start,start+args.epochs):
            model.train(); loss_sum=0.; count=0
            for x,y in trainloader:
                x,y=x.to(device),y.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss=criterion(model(x),y); loss.backward(); optimizer.step()
                loss_sum+=loss.item()*len(y); count+=len(y)
            metrics=evaluate(model,testloader,device)
            writer.add_scalar('loss/train',loss_sum/count,epoch)
            writer.add_scalar('loss/test',metrics['loss'],epoch)
            writer.add_scalar('accuracy/test',metrics['accuracy'],epoch)
            print(json.dumps({'epoch':epoch+1,'dataset':name,'train_loss':loss_sum/count,'test':metrics}))
            torch.save({'architecture':'cnn_relu_v1','dataset':name,'classes':CLASSES,
                        'preprocess':'resize_32_totensor','epoch':epoch+1,
                        'model':model.state_dict(),'optimizer':optimizer.state_dict(),
                        'torch_rng_state':torch.get_rng_state(),'python_rng_state':random.getstate(),
                        'cuda_rng_state':torch.cuda.get_rng_state_all() if device.type=='cuda' else None},checkpoint)
    finally:
        writer.close()
    print('checkpoint:',checkpoint)

if __name__=='__main__': main()
