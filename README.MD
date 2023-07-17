# DDSP-HiFiGAN：基于PC-DDSP和nsf-HiFiGAN的声码器

DDSP-HiFiGAN是一个基于[PC-DDSP](https://github.com/yxlllc/pc-ddsp)和nsf-[HiFiGAN](https://github.com/jik876/hifi-gan)的声码器，由DSPGAN所启发，同时具有PC-DDSP无限变调的优势和nsf-HiFiGAN高音质的优势。

# 使用方法

## 1. 安装依赖
执行以下命令以安装依赖。
```bash
pip install -r requirements.txt 
```
## 2. 获取预训练模型

### PC-DDSP模型
可以依照[pc-ddsp](https://github.com/yxlllc/pc-ddsp)来训练一个ddsp模型, 把训练好的模型权重文件`model_xxx.pt`和配置文件`config.yaml`一同放到目录`pretrained/ddsp`下。

### nsf-HiFiGAN模型

可以下载openvpi训练的[社区声码器](https://github.com/openvpi/vocoders/releases)，使用时请遵循相关协议。把配置文件`config.json`、鉴别器权重`do_xxx`、生成器权重`g_xxx`一同放到目录`pretrained/hifigan`下。

## 3. 数据预处理
把用于训练的wav音频数据放到`data/train/audio`下，用于验证的wav音频数据放到`data/val/audio`下，再执行以下命令：
```bash
python preprocess.py --resample
```

## 4. 微调nsf-HiFiGAN
目前仅支持从已有的权重微调nsf-HiFiGAN，暂不支持从头开始训练或与ddsp级联训练。
```bash
python train.py --inverse_aug
```

## 5. 可视化微调效果
```bash
tensorboard --logdir=pretrained/hifigan/logs
```

## 6. 从wav中推理
如果想要突破声码器的音域限制，那么就可以-ak参数。
```bash
python inference.py -dm <ddsp model path> -hm <hifigan model path> -i <input.wav> -o <output.wav> -k <keychange (semitones)> -ak <adaptive_key (semitones)>
```
# 注意事项
仅在单卡下（autodl的3090）测试过，无法保证多卡训练不出错，并且单卡一般已经足够微调。
