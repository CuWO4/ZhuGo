# 部署要点

- JIT 加速
  - 关于 JIT 无法用于训练, TorchScript 虽然可以正确反向传播和训练, 但是因为无法捕获动态路径, 故即使 eval 也无法使 BN 和 Dropout 进入推理路径, 仍然保持训练行为. 故 TorchScript 训练出的模型无法进入推理模式, 只可用于推理部署. 另一方面 TorchScript 训练速度增益不明显.
  - JIT eval 性能统计:

    | 模式 | epoch/sec | sec/epoch |
    | :--: | :--: | :--: |
    | baseline (原生pytorch + eval + no_grad) | 6.8 | 0.15 |
    | TorchScript | 7.2 | 0.14 |
    | TorchScript + export 注解 | 7.2 | 0.14 |
    | TorchScript + trace | 8.2 | 0.12 |

- eval() 模式, with torch.no_grad()
