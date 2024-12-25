# 🚀 Python 高性能推流工具

一个强大的多功能推流工具，支持多路流同时推送！ 🎯

## ✨ 功能特点

- 🔥 支持同时推送20+路流
- 📺 支持多种输入源：
  - 📡 RTMP流转推
  - 🎬 本地MP4视频
  - 🌐 远程MP4视频
  - 📹 FLV视频
  - 🖥️ 本地桌面录屏
- 🎮 支持单独控制每个流的开启和关闭
- 📝 支持从配置文件批量读取流列表
- 🔄 支持视频循环播放
- ⚡ 高性能，低资源占用

## 🛠️ 安装要求

1. 🐍 Python 3.7+
2. 🎥 FFmpeg（需要预先安装）

### 📦 安装依赖包：
```bash
pip install -r requirements.txt
```

## 🎯 使用方法

### 1️⃣ 配置文件设置

编辑 `config.json` 文件，配置你的推流任务：

```json
{
    "enabled": true,
    "streams": [
        {
            "enabled": true,
            "stream_id": "video_list_1",
            "stream_type": "video_list",
            "output_url": "rtmp://localhost/live/stream",
            "loop": true,
            "video_list": [
                "/path/to/video1.mp4",
                "/path/to/video2.mp4"
            ]
        }
    ]
}
```

### 2️⃣ 启动程序

```bash
python stream_pusher.py
```

### 3️⃣ 停止程序

按 `Ctrl+C` 优雅地停止所有推流 🛑

## 🎨 支持的流类型

### 📺 视频列表推流
```json
{
    "stream_type": "video_list",
    "video_list": ["video1.mp4", "video2.mp4"]
}
```

### 📡 RTMP流转推
```json
{
    "stream_type": "rtmp",
    "input_source": "rtmp://source.com/live/stream1"
}
```

### 🎬 本地视频推流
```json
{
    "stream_type": "local_video",
    "input_source": "/path/to/video.mp4"
}
```

### 🖥️ 屏幕录制推流
```json
{
    "stream_type": "screen",
    "input_source": "screen"
}
```

### 🌐 远程视频推流
```json
{
    "stream_type": "remote_video",
    "input_source": "http://example.com/video.mp4"
}
```

## ⚙️ 配置说明

- `enabled`: 全局或单个流的开关 (true/false)
- `stream_id`: 流的唯一标识符
- `stream_type`: 流类型
- `input_source`: 输入源地址
- `output_url`: 推流目标地址
- `loop`: 是否循环播放
- `video_list`: 视频列表（仅用于 video_list 类型）

## 🚨 注意事项

1. ✅ 确保系统已安装 FFmpeg
2. 🌐 确保有足够的网络带宽
3. 💻 确保有足够的系统资源
4. 📁 推送本地文件时，确保文件路径正确
5. 🔒 推送 RTMP 流时，确保有正确的访问权限

## 🤝 贡献

欢迎提交 Issues 和 Pull Requests！ 让我们一起把这个工具做得更好！ 🎉

## 📄 许可证

MIT License 📝