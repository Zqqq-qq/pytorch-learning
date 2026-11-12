"""独立运行整理版 Notebook，验证梯度、checkpoint 与因果性等核心行为。"""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import nbformat
from nbclient import NotebookClient
import torch

ROOT=Path(__file__).resolve().parents[1]

def main():
    for entry in json.loads((ROOT/'originals/manifest.json').read_text()):
        assert hashlib.sha256((ROOT/entry['archive']).read_bytes()).hexdigest()==entry['sha256']
    for path in sorted((ROOT/'notebooks').glob('*.ipynb')):
        notebook=nbformat.read(path,as_version=4); nbformat.validate(notebook)
        with tempfile.TemporaryDirectory() as tmp:
            # 每章一个新内核、独立目录，排除跨笔记本的隐藏变量依赖。
            NotebookClient(notebook,timeout=180,kernel_name='python3',
                           resources={'metadata':{'path':tmp}}).execute()
        print('PASS',path.name,flush=True)
    spec=importlib.util.spec_from_file_location('gpt_reference',ROOT/'reference/gpt_from_original.py')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    torch.manual_seed(42)
    config=module.GPTConfig(block_size=8,vocab_size=16,n_layer=1,n_head=2,n_embd=16)
    model=module.GPT(config)
    x=torch.randint(0,16,(2,5)); logits,loss=model(x,x)
    assert logits.shape==(2,5,16) and torch.isfinite(loss)
    loss.backward(); model.eval()
    generated=model.generate(x,max_new_tokens=2,top_k=4)
    assert generated.shape==(2,7)
    print('PASS GPT 微型配置前向/反向/生成；未验证预训练权重')

if __name__=='__main__': main()
