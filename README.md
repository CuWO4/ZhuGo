# ZHU GO

基于蒙特卡洛树搜索的围棋机器人.

启发:
<https://deepmind.google/discover/blog/alphazero-shedding-new-light-on-chess-shogi-and-go/>
<https://github.com/maxpumperla/deep_learning_and_the_game_of_go>

![猪](docs/pic/zhu.jpg)
\* 这是猪, 他没怎么样, 只是他很可爱想给你们看看.

## 使用

```shell
python main.py -c CONF 
```

将依照配置文件启动棋局. 配置文件可修改棋局信息(如棋盘尺寸, 贴目等), 双方代理(如人类棋手, 传统随机蒙特卡洛树搜索算法bot等), 使用的 GUI等等. 具体可以参考 `conf/main/`.
