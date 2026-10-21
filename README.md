# Cowboy Outfits 目标检测（CodaLab 33573）

基于《动手学深度学习》目标检测一章的实战比赛：检测牛仔穿搭照片中的
5 类目标（belt / sunglasses / boot / cowboy_hat / jacket），COCO 格式标注，
3062 张训练图（图片数据不入库，见下方复现步骤）。

## 方法

`cowboy-outfits-pytorch.ipynb`（d2l 教材风格）：

- **模型**：ultralytics **YOLOv5su**（YOLOv5-s + v8 anchor-free 检测头），
  加载 COCO 预训练权重迁移学习，imgsz=1024 兼顾小目标（belt 在低分辨率下只剩几十像素）
- **类别不平衡**：验证集按类别分层切分保证每类有样本；极端稀有的 belt
  （全集仅 25 个框）做图像级过采样 ×6
- **评估**：COCO 官方 COCOeval，与 Faster R-CNN 旧基线同口径可比
- **TTA**：`predict(augment=True)` 水平翻转 + 多尺度
- **踩坑记录**（notebook 内有注释）：
  - image_id 是 uint64 可能超 int64，COCOeval 静默转 float64 丢精度 → 映射成连续小整数
  - ultralytics 的列表推理源不分批，999 张图一个 batch 必 OOM → 自行分块
  - Windows DataLoader spawn 子进程无法 import notebook 里的类 → Dataset 拆独立 .py（见 `archive/`）

## 结果

| 阶段 | mAP@[.5:.95] | 备注 |
|------|-------------|------|
| Faster R-CNN（6 项优化 + TTA） | **0.448** | 旧基线，代码在 `archive/fasterrcnn/` |
| YOLOv5su（12 epoch @1024） | 0.328* | ultralytics 内部指标，COCOeval 同口径复评见 notebook |

*两种口径不可直接对比，待 COCOeval 复评后更新。

## 文件说明

```
├── cowboy-outfits-pytorch.ipynb      # 主 notebook（数据→YOLO 标签→训练→评估→提交）
├── download_dataset.py               # kagglehub 数据集下载器（需 KAGGLE_API_TOKEN）
├── cowboy.yaml                       # ultralytics 数据集配置
├── train.json                        # COCO 格式标注
├── valid.csv / test.csv              # 公开榜 / 决赛图片清单
├── answer.json / sample_answer.zip   # 提交文件（CodaLab 要求 zip 内含 answer.json）
├── _smoke_yolov5.py                  # COWBOY_QUICK=1 全链路冒烟测试
├── _probe_oom.py                     # 训练显存峰值探针（b4/b6/b8 实测）
├── _viz_submission.py                # 提交结果可视化
└── archive/fasterrcnn/               # Faster R-CNN 旧基线（mAP 0.448）
```

## 复现步骤

1. `KAGGLE_API_TOKEN=<token> python download_dataset.py` 下载数据集，
   解压出 `images/` 并放好 `train.json`；
2. 从镜像下载预训练权重 `yolov5su.pt`（地址见 notebook）；
3. 依次运行 notebook：自动完成 COCO→YOLO 标签转换、训练、COCOeval 评估、生成提交文件。

## 致谢

方法论参考《动手学深度学习》（[d2l-zh](https://github.com/d2l-ai/d2l-zh)）目标检测章节；
数据来自 GluonCV 团队的 CowboySuit 数据集。
