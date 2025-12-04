# MultiModal模型昇腾迁移优化技术报告

## 📊 评测结果

| 评测指标 | 平均得分 |
|---------|---------|
| 峰值显存得分 | 100 |
| Prefill时延得分 | 162.959 |
| Decode时延得分 | 122.4649 |
| **总分** | **128.4747** |

## 🎯 优化模型

本项目针对以下两个MultiModal模型进行了昇腾NPU适配与性能优化：

1. **deepseek-ai/Janus-Pro-7B**
2. **Qwen/Qwen2-VL-2B-Instruct**

---

## 🔧 核心优化技术

### 使用mint算子替代ops算子
如：
```python
# 原conv3d
output = self.conv3d(input, self.weight)
# 替换为使用mint的conv3d加速, 省去编译算子时间
output = mint.nn.functional.conv3d(input, self.weight, bias=None, stride=self.stride, padding=0, dilation=self.dilation, groups=self.groups)
```
**收益**：mint算子针对昇腾NPU深度优化，减少算子调度开销，提升计算效率。

### 使用ops替代索引操作
如：
```python
# 原索引操作
x1 = x[..., : x.shape[-1] // 2]
x2 = x[..., x.shape[-1] // 2 :]
# 使用ops算子替代
x1,x2 = ops.split(x, x.shape[-1] // 2,dim=-1)
```
**收益**：避免动态索引带来的性能损失。

### RMSNorm使用融合算子加速
```python
# 原RMSNorm算子
input_dtype = hidden_states.dtype
hidden_states = hidden_states.to(mindspore.float32)
variance = ops.mean(hidden_states.pow(2), -1, keepdim=True)
hidden_states = hidden_states * ops.rsqrt(variance + self.variance_epsilon)
return self.weight * hidden_states.to(input_dtype)

#融合RMSNorm算子
return F.rms_norm(hidden_states, self.weight, self.variance_epsilon)
```
**收益**：推理时使用融合rms_norm，减少计算开销。
