# 性能问题

```prof
         423773840 function calls (415712835 primitive calls) in 233.125 seconds

   Ordered by: cumulative time
   List reduced from 437 to 20 due to restriction <20>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000  233.125  233.125 G:\code\projects\Go\ZhuGo\test.py:18(test_func)
        1    0.972    0.972  232.439  232.439 G:\code\projects\Go\ZhuGo\test.py:10(worker)
     5000    0.139    0.000  210.522    0.042 G:\code\projects\Go\ZhuGo\ai\encoder\zhugo_encoder.py:53(encode)
     5000   24.319    0.005   92.257    0.018 G:\code\projects\Go\ZhuGo\ai\encoder\zhugo_encoder.py:120(encode_qi_after_play)
  5415000    8.336    0.000   69.591    0.000 G:\code\projects\Go\ZhuGo\go\goboard.py:252(does_move_violate_ko)
     5000   19.122    0.004   68.160    0.014 G:\code\projects\Go\ZhuGo\ai\encoder\zhugo_encoder.py:92(encode_valid_move)
  3610000    6.175    0.000   66.491    0.000 G:\code\projects\Go\ZhuGo\go\goboard.py:260(is_valid_move)
14440000/7220000   24.310    0.000   60.509    0.000 G:\Python\lib\copy.py:128(deepcopy)
  7220000    5.033    0.000   38.842    0.000 G:\code\projects\Go\ZhuGo\go\goboard.py:112(__deepcopy__)
     5000    2.045    0.000   29.809    0.006 G:\code\projects\Go\ZhuGo\ai\encoder\zhugo_encoder.py:153(encode_ko)
  1805000    3.428    0.000   25.557    0.000 G:\code\projects\Go\ZhuGo\go\goboard.py:224(apply_move)
     5000    0.010    0.000   20.754    0.004 G:\code\projects\Go\ZhuGo\ai\zhugo.py:122(__call__)
     5000    0.048    0.000   20.745    0.004 G:\code\projects\Go\ZhuGo\ai\zhugo.py:116(forward)
420000/15000    0.372    0.000   20.602    0.001 G:\Python\lib\site-packages\torch\nn\modules\module.py:1507(_wrapped_call_impl)
420000/15000    0.773    0.000   20.584    0.001 G:\Python\lib\site-packages\torch\nn\modules\module.py:1513(_call_impl)
45000/15000    0.446    0.000   20.505    0.001 G:\Python\lib\site-packages\torch\nn\modules\container.py:215(forward)
 14440000    8.334    0.000   15.539    0.000 G:\code\projects\Go\ZhuGo\go\goboard.py:70(get)
  1805000    2.246    0.000   12.755    0.000 G:\code\projects\Go\ZhuGo\utils\eye_identifier.py:10(is_point_an_eye)
 14440000    8.928    0.000   11.389    0.000 G:\Python\lib\copy.py:243(_keep_alive)
  7220000    4.011    0.000   10.621    0.000 G:\code\projects\Go\ZhuGo\cboard\cboard.py:27(__deepcopy__)

`decorated_func` takes 39117.04ms
```

三大困局:

* 编码过于耗时, 代价不可接受.

* 使用 C/C++ 重写游戏引擎过于复杂, 代价不可接受.

* 树搜索不利于多进程编码-推导, 或需复杂的调度算法.

## 调度算法 1

`核数 - 1` 个进程中, 中主进程不断将 ucb 最大的未被编码局面推入任务队列, 其余从进程从队列中获取任务并进行编码; 主进程选择ucb最大的局面, 忙等待到局面被编码好, 并传送给 GPU 进行推理.

* 如何定义不同从局面之间的 ucb 谁更大?

* 如何维护 "未被编码过的最大 ucb 动作"?

* 如何同步 ucb 信息?

* 如何提高编码利用率? 大量未被使用的编码张量是否会导致内存消耗陡增?

## 调度算法 2

同时编码当前 ucb 最大的 `核数` 个局面, 并传送给 GPU 处理.

* 如何定义不同局面之间的 ucb 谁更大?

* 如何维护 "未被编码过的最大 ucb 动作"?

* 过于集中的 ucb 策略是否会使不良动作影响估计精度; 过于宽泛的 ucb 策略是否会使搜索深度过低, 且无法集中于关键动作?

## 调度算法 3

编码局面时, 维护一个 "虚 ucb", 开始被初始化为 ucb. 尝试编码虚 ucb 最大的子局面, 完成后假设子局面的 V 与父局面保持一致, 增加虚访问次数, 更新虚 ucb; 当被编码局面确实被访问时, 通知父节点真实 V, 各父节点修正访问次数和 q,  以将虚 ucb 与真实 ucb 保持一致. 当沿虚 ucb 搜索来到已被标记可编码, 但还没有进行计算的节点(虚节点)时, 不进行任何操作, 只增加虚访问次数. 一次编码若干批量的虚节点, 编码完成后等待推理进程发送信号, 需要编码未被编码的节点时, 再继续编码.

搜索中总是会产生大量探索次数少于10次的局面, 甚至应该说绝大多数都是这样的局面, 在这些局面上执行虚调度, 可以取得良好的效果, 充分将有潜力的局面纳入考虑. 此外, 虚节点不进行操作也可以保证内存不会膨胀. 一次编码一个批量保证了实际 ucb 最大的节点不会被滞后更新导致迟迟无法被分配编码资源, 从而浪费 GPU 时间.

只要神经网络 V 输出是充分有预见性的, 那么子节点产生的 V 差异可以控制在较小范围内, 就可以保证调度近似前提子节点和父节点 V 相同, 从而提高调度命中率. 也就意味着该调度算法在训练初期会异常缓慢.

此外, zero 的 ucb 探索项 $\frac{\sqrt N}{n + 1}$ 比之于传统探索项 $ \sqrt{\frac{\ln N}{n + 1}} $ 更鼓励探索和树的扩展, 从而使得虚扩展阶段频繁扩展到虚节点概率更低, 每个批次实际编码局面数不会远远小于批次数.

## 多进程编码方案 1

将编码改为多进程.

## 调度算法 4

使用类似双缓冲的两个进程, 分别执行 `预测(选择当前最大ucb) -> 编码 -> 提交 -> 更新` 流程, 由于有两个进程在进行提交, 计算任务会被存放在 GPU 的任务队列上, 从而免去了编码, GPU 传送, 更新带来的 GPU 等待.

由于 Python 沟槽的共享内存和 GIL, 所以还需要第三个进程来进行辅助, 负责拥有树, 并相应两个进程的请求.

```text
      Master        Process A       Process B          GPU
 ==============================================================
      | | |               | | |               | | |               |
      | P +--- success ---+ P |               | | |               |
      | P +----success--- | E | --------------+ P |               |
      | | |               | S +---- Task 1 -- | E | -----------> ===
      | | |               | | |               | S + -- Task 2 -> |||
      | | |               | | |               | | |              |||
      | | |               | | |               | | |              ||| Task 1
      | | |               | | |               | | |              |||
      | U <---------------+ U <---- Task 1 -- | | | -----------+ ===
      | P <---- fail -----+ P |               | | |              |||
      | | |               | P |               | | |              |||
      | | |               | P |               | | |              ||| Task 2
      | | |               | P |               | | |              |||
      | | |               | P |               | | |              |||
      | U <-------------- | P | --- Task 2 ---+ U <------------+ ===
      | P +--- success ---> P |               | P |               |
      | P +--- success -- | E | --------------+ P |               |
      | | |               | S +---- Task 3 -- | E | -----------> ===
      | | |               | | |               | S + -- Task 4 -> |||
      | | |               | | |               | | |              |||
      | | |               | | |               | | |              ||| Task 3
      | | |               | | |               | | |              |||
      | | |               | | |               | | |              |||
      | U <---------------+ U <---- Task 3 -- | | | -----------+ ===
      | P +----success----+ P |               | | |              |||
      | | |               | E |               | | |              |||
      | | |               | S +---- Task 5 -- | | | -----------> ||| Task 4
      | | |               | | |               | | |              |||
      | | |               | | |               | | |              |||
      | U <-------------- | | | --- Task 4 ---+ U <------------+ ===
      | P <---- fail ---- | | | --------------+ P |              |||
      | | |               | | |               | P |              |||
      | | |               | | |               | P |              ||| Task 5
        .                   .                   .                 .
        .                   .                   .                 .
        .                   .                   .                 .
```

问题: 当两个进程访问同一个节点时, 应该如何处理?

* **解决方法 1**
   当节点已经进入计算流程时, 选择 ucb 第二大的选择

   **错误**. 因为首先选择 ucb 是一个从根到底的操作, 当前节点的 ucb 第二大不意味着整体看第二值得选. 在某些只有唯一选的局面时, 会在初期探索时引入巨大的偏差.

* **解决方法 2**

   ```python
   if node.lock.acquire(block = False):
      # encode, calculate, update
      node.lock.release()
   else:
      with node.lock:
         pass
   ```

   低效. 完成等待后两个进程又会面临相同的局面, 于是两个进程总是处于访问同一个节点的状态.

* **解决方法 3**
   当预测到达已进入计算流程的节点时, 为路径增加一个伪访问次数, 然后再次从根开始, 直到决策树因为伪访问分叉. 在更新完释放锁后, 清楚当前路径中的所有伪访问次数(必然只在该条到根路径上, 且每个节点上伪访问次数相等)

   **问题**: 在某些情况下, 特别是决策树高度未被探索的时候, 大量增加伪访问次数仍然会导致解决方法 1 中的不稳定.

   **可能解决**: 当且仅当伪访问次数不超过真实访问次数的某个比例时, 继续使用伪访问次数尝试分叉, 否则阻塞在当前节点, 直到另一进程访问完毕伪访问被清空.

   如此一来就控制了伪访问给结果带来的偏差, 与此同时, 因为 ucb 公式鼓励探索, 增加少量的伪访问就可以完成分叉, 保证了GPU利用率和性能.

   ```python
   if node.lock.acquire(block = False):
      # encode, calculate, update
      node.lock.release()
      # clear pseudo visit
   else:
      if node.pseudo_visit < k * node.visit: # for say k = 0.2
         # add pseudo visit to path
      else:
         with node.lock:
            pass
   ```
