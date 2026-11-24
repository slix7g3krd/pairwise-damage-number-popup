# pairwise-damage-number-popup

## 领域

游戏、图形与多媒体

## 主题

伤害飘字投影与对象池

## 说明

当前相机投影、池扩容与固定种子偏移。

初始状态下 `gameplay/damage_number_popup.py` 中所有函数抛出 `NotImplementedError`，测试全部失败。
修复实现后运行：

```bash
python -m pytest -q
```

期望全部通过。
