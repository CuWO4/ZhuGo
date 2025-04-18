# Compress Other Train Set Format to ZhuGo Train Set Format BGTF

## Leela Go (only supported format so far)

1. prepare train set from <leela.online-go.com/training> or other channels.

1. unzip them and rename them with `.leela` suffix, then place them in a working directory. (if you do not want to unzip since the file size is unaffordable, write a script to dynamically unzip)

1. execute

```shell
python compress.py --type leela --level [LEVEL] [WORK_DIRECTORY] [OUTPUT_DIRECTORY]
```

, the script will compress each `[WORK_DIRECTORY]/*.leela` to `[OUTPUT_DIRECTORY]/*.bgtf.zstd`, which can be paused with ZhuGo dataloader.

`[LEVEL]` specify the compress level, which is 3 by default.

## Other Formats

TODO
