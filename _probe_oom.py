"""显存峰值探针：实测 yolov5su @1024 在 batch 4/6/8 下的真实显存峰值 + TTA 预测开销。

用途：OOM 定位与 batch 上限验证。每个探针独立 try/except，OOM 打印 traceback 后继续。
"""
import os, sys, json, gc
from collections import Counter

os.environ.setdefault('PYTHONUTF8', '1')
PROJ = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJ)
VENDOR = os.path.join(PROJ, 'vendor')
if os.path.isdir(os.path.join(VENDOR, 'numpy')):
    sys.path.insert(0, VENDOR)

import torch
assert torch.cuda.is_available()
from ultralytics import YOLO


def mem(tag):
    a = torch.cuda.memory_allocated() / 2**30
    r = torch.cuda.memory_reserved() / 2**30
    m = torch.cuda.max_memory_allocated() / 2**30
    print(f'>>> [{tag}] allocated {a:.2f}G | reserved {r:.2f}G | peak {m:.2f}G', flush=True)
    torch.cuda.reset_peak_memory_stats()


def run(tag, fn):
    print(f'===== {tag} =====', flush=True)
    try:
        fn()
    except torch.OutOfMemoryError as e:
        print(f'>>> OOM: {str(e)[:200]}', flush=True)
        import traceback
        traceback.print_exc()
        mem(tag + ' AFTER OOM')
        gc.collect(); torch.cuda.empty_cache()
    except Exception:
        import traceback
        traceback.print_exc()
        gc.collect(); torch.cuda.empty_cache()


# ---- 探针用 mini 数据（真实图前 64 张）----
train_lines = [l for l in open('train_yolo.txt', encoding='utf-8').read().split('\n') if l]
val_lines = [l for l in open('val_yolo.txt', encoding='utf-8').read().split('\n') if l]
open('_probe.txt', 'w', encoding='utf-8').write('\n'.join(train_lines[:64]) + '\n')
open('_probe_val.txt', 'w', encoding='utf-8').write('\n'.join(val_lines[:8]) + '\n')
names = ["belt", "sunglasses", "boot", "cowboy_hat", "jacket"]
open('_probe.yaml', 'w', encoding='utf-8').write(
    f'train: {PROJ}\\_probe.txt\nval: {PROJ}\\_probe_val.txt\nnc: 5\nnames: {names}\n')


def probe_train(batch):
    def _run():
        m = YOLO('yolov5su.pt')
        m.train(data='_probe.yaml', epochs=1, imgsz=1024, batch=batch, device=0,
                workers=0, cache=False, project=PROJ, name='_probe_oom',
                exist_ok=True, plots=False, verbose=False,
                cos_lr=True, close_mosaic=1, patience=2, seed=0)
        mem(f'su batch{batch} @1024 trained')
        del m
        gc.collect(); torch.cuda.empty_cache()
    return _run


def probe_predict(paths, tag):
    def _run():
        m = YOLO(os.path.join(PROJ, 'yolov5_cowboy', 'weights', 'best.pt'))
        r = m.predict(paths, imgsz=1024, conf=0.05, iou=0.7,
                      augment=True, half=True, device=0, verbose=False)
        mem(tag)
        del m, r
        gc.collect(); torch.cuda.empty_cache()
    return _run


for b in (4, 6, 8):
    run(f'train su batch{b} @1024', probe_train(b))
run('predict TTA x84(val) @1024', probe_predict(val_lines[:84], 'predict tta val84'))
print('\n[probe] DONE', flush=True)
