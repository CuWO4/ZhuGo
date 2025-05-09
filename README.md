# ZHU GO

基于深度学习和蒙特卡洛树搜索的围棋机器人.

启发:

- Alpha Zero, Deep Mind <https://deepmind.google/discover/blog/alphazero-shedding-new-light-on-chess-shogi-and-go/>
- Deep Learning and the Game of Go, Kevin Ferguson and Max Pumperla <https://github.com/maxpumperla/deep_learning_and_the_game_of_go>
- Leela Zero, Leela Zero Team <https://github.com/leela-zero/leela-zero>

![猪](docs/pic/zhu.jpg)
\* 这是猪, 他没怎么样, 只是他很可爱想给你们看看.

## Demo

TODO

## Build

在根目录执行

```shell
python setup.py
```

以编译 C 扩展.

## Usage

```shell
python main.py play -c CONF
```

将依照配置文件启动棋局. 配置文件可修改棋局信息(如棋盘尺寸, 贴目等), 双方代理(如人类棋手, 传统随机蒙特卡洛树搜索算法bot等), 使用的 GUI等等. 具体可以参考 `conf/main/`.

```shell
python main.py create -c CONF -p PATH
```

将按照配置文件创建模型. 具体可以参考 `conf/model/`.

```shell
python -O main.py -m [MODEL ROOT] -d [DATASET ROOT] -c CONF
```

将开启训练, 具体可参考 `conf/train/`.

使用

```shell
python main.py --help
```

查看更多细节.
