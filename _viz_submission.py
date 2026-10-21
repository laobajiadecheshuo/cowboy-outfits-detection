"""把 answer.json 的预测结果画到图上，输出一张 6 宫格预览 PNG。不依赖 torch。"""
import os, json, zipfile
from collections import Counter, defaultdict

os.environ.setdefault('PYTHONUTF8', '1')
os.environ.setdefault('MPLBACKEND', 'Agg')
PROJ = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJ)

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from pycocotools.coco import COCO
import pandas as pd

# 类别映射（与 notebook 一致：loadCats 顺序 = 提交时的类别索引依据）
coco = COCO('train.json')
cats = coco.loadCats(coco.getCatIds())
id2name = {c['id']: c['name'] for c in cats}
palette = ['red', 'blue', 'green', 'orange', 'purple']
name2color = {c['name']: palette[i % 5] for i, c in enumerate(cats)}

# image_id -> 文件路径（valid.csv 里 id 按字符串读，防大整数丢精度）
df = pd.read_csv('valid.csv', dtype={'id': 'object'})
id2path = {row['id']: os.path.join(PROJ, 'images', row['file_name'])
           for _, row in df.iterrows()}

answer = json.load(open('answer.json', encoding='utf-8'))
by_img = defaultdict(list)
for r in answer:
    by_img[r['image_id']].append(r)

# ---- 文本统计 ----
cnt = Counter(id2name[r['category_id']] for r in answer)
scores = [r['score'] for r in answer]
print(f'图片数: {len(by_img)} / {len(df)} | 预测框总数: {len(answer)} | '
      f'平均置信度: {sum(scores) / len(scores):.3f}')
print('各类别框数:', dict(cnt.most_common()))

# ---- 挑图：每个类别取其最高置信度检测所在的图（去重），不足 6 张用框数最多的补 ----
picked, seen = [], set()
for name in [c['name'] for c in cats]:
    best = max((r for r in answer if id2name[r['category_id']] == name),
               key=lambda r: r['score'], default=None)
    if best and best['image_id'] not in seen:
        seen.add(best['image_id'])
        picked.append(best['image_id'])
for img_id in sorted(by_img, key=lambda k: -len(by_img[k])):
    if len(picked) >= 6:
        break
    if img_id not in seen:
        seen.add(img_id)
        picked.append(img_id)
picked = picked[:6]
print('picked image_ids:', picked)

# ---- 画 2x3 六宫格 ----
fig, axes = plt.subplots(2, 3, figsize=(16, 11))
for ax, img_id in zip(axes.ravel(), picked):
    ax.imshow(Image.open(id2path[str(img_id)]))
    ax.set_title(f'image_id {img_id} | {len(by_img[img_id])} boxes', fontsize=10)
    ax.axis('off')
    for r in sorted(by_img[img_id], key=lambda r: -r['score']):
        name = id2name[r['category_id']]
        x1, y1, x2, y2 = r['bbox']
        color = name2color[name]
        ax.add_patch(mpatches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                        linewidth=2, edgecolor=color, facecolor='none'))
        ax.text(x1, max(y1 - 4, 0), f'{name} {r["score"]:.2f}', color='white',
                fontsize=8, weight='bold', va='bottom',
                bbox=dict(facecolor=color, alpha=0.85, lw=0, pad=1))
plt.tight_layout()
out = os.path.join(PROJ, '_preview_predictions.png')
plt.savefig(out, dpi=110, bbox_inches='tight')
print('saved:', out)

# 顺带确认 zip 内容
with zipfile.ZipFile('sample_answer.zip') as zf:
    print('zip 内文件:', zf.namelist())
