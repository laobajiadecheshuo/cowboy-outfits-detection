"""冒烟测试：顺序执行 notebook 的所有 code cell（过滤 % 行级 magic），验证全链路。

用法：COWBOY_QUICK=1 python _smoke_yolov5.py
必须带 __main__ 保护：Windows spawn 的 DataLoader worker 会重新 import 本脚本，
没有保护的话每个 worker 都会把整个 notebook 重跑一遍。
"""
import os, sys, json

os.environ.setdefault('COWBOY_QUICK', '1')
os.environ.setdefault('PYTHONUTF8', '1')
os.environ.setdefault('MPLBACKEND', 'Agg')

NB = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cowboy-outfits-pytorch.ipynb')


def main():
    with open(NB, encoding='utf-8') as f:
        nb = json.load(f)
    ns = {'__name__': '__main__'}
    n = 0
    for i, cell in enumerate(nb['cells']):
        if cell.get('cell_type') != 'code':
            continue
        src = cell['source']
        if isinstance(src, list):
            src = '\n'.join(src)
        # 过滤行级 magic / shell 转义（如 %matplotlib inline）
        src = '\n'.join(ln for ln in src.split('\n')
                        if not ln.lstrip().startswith(('%', '!')))
        if not src.strip():
            continue
        n += 1
        print(f'\n########## [smoke] cell {i} (#{n}) ##########', flush=True)
        try:
            exec(compile(src, f'cell_{i}', 'exec'), ns)
        except Exception:
            import traceback
            traceback.print_exc()
            print(f'\n[smoke] FAILED at cell {i}', flush=True)
            sys.exit(1)
    print('\n[smoke] ALL CELLS PASSED', flush=True)


if __name__ == '__main__':
    main()
