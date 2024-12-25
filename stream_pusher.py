import subprocess
import threading
import time
import os
import json
from typing import Dict, List, Optional
import mss
import cv2
import numpy as np
from queue import Queue
from dataclasses import dataclass
from datetime import datetime


@dataclass
class StreamConfig:
    stream_id: str
    input_source: str  # 可以是RTMP URL、本地文件路径或'screen'
    output_url: str
    stream_type: str  # 'rtmp', 'local_video', 'remote_video', 'screen', 'video_list'
    loop: bool = False
    video_list: List[str] = None  # 用于video_list类型，存储视频文件路径列表


class StreamPusher:
    def __init__(self):
        self.streams: Dict[str, subprocess.Popen] = {}
        self.stream_configs: Dict[str, StreamConfig] = {}
        self.running = True
        self._lock = threading.Lock()
        self.ffmpeg_path = self._get_ffmpeg_path()

    def _get_ffmpeg_path(self) -> str:
        # 获取当前脚本所在目录
        current_dir = os.path.dirname(os.path.abspath(__file__))

        # 检查同目录下的ffmpeg可执行文件
        if os.name == 'nt':  # Windows
            ffmpeg_path = os.path.join(current_dir, 'ffmpeg.exe')
        else:  # Linux/Mac
            ffmpeg_path = os.path.join(current_dir, '../CarStream/ffmpeg')

        # 如果同目录下存在ffmpeg，使用它，否则使用系统ffmpeg
        if os.path.exists(ffmpeg_path):
            return ffmpeg_path
        return 'ffmpeg'  # 使用系统PATH中的ffmpeg

    def start_video_list_stream(self, stream_id: str, video_list: List[str], output_url: str,
                                loop: bool = True) -> bool:
        """
        启动一个视频列表推流任务
        :param stream_id: 流ID
        :param video_list: 视频文件路径列表
        :param output_url: 推流目标地址
        :param loop: 是否循环播放列表
        :return: 是否成功启动
        """
        config = StreamConfig(
            stream_id=stream_id,
            input_source="",  # 这里不使用input_source字段
            output_url=output_url,
            stream_type="video_list",
            loop=loop,
            video_list=video_list
        )
        return self.start_stream(config)

    def start_stream(self, config: StreamConfig) -> bool:
        with self._lock:
            if config.stream_id in self.streams:
                print(f"Stream {config.stream_id} already exists")
                return False

            try:
                if config.stream_type == 'screen':
                    thread = threading.Thread(target=self._push_screen,
                                              args=(config,))
                    thread.daemon = True
                    thread.start()
                    self.streams[config.stream_id] = thread
                elif config.stream_type == 'video_list':
                    thread = threading.Thread(target=self._push_video_list,
                                              args=(config,))
                    thread.daemon = True
                    thread.start()
                    self.streams[config.stream_id] = thread
                else:
                    command = self._build_ffmpeg_command(config)
                    process = subprocess.Popen(
                        command,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE
                    )
                    self.streams[config.stream_id] = process
                self.stream_configs[config.stream_id] = config
                return True
            except Exception as e:
                print(f"Error starting stream {config.stream_id}: {str(e)}")
                return False

    def _push_video_list(self, config: StreamConfig):
        """
        处理视频列表的推流
        """
        while config.stream_id in self.streams:
            for video_path in config.video_list:
                if not os.path.exists(video_path):
                    print(f"Video file not found: {video_path}")
                    continue

                if config.stream_id not in self.streams:
                    break

                command = [
                    self.ffmpeg_path,
                    '-re',  # 以原始帧率读取
                    '-i', video_path,
                    '-c:v', 'libx264',
                    '-preset', 'veryfast',
                    '-c:a', 'aac',
                    '-f', 'flv',
                    config.output_url
                ]

                try:
                    process = subprocess.Popen(
                        command,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE
                    )
                    # 等待当前视频推流完成或者被中断
                    process.wait()
                except Exception as e:
                    print(f"Error pushing video {video_path}: {str(e)}")
                    if process:
                        process.terminate()
                        process.wait(timeout=5)

            if not config.loop:
                break

    def stop_stream(self, stream_id: str) -> bool:
        with self._lock:
            if stream_id not in self.streams:
                return False

            try:
                if isinstance(self.streams[stream_id], threading.Thread):
                    # 对于屏幕录制线程，我们需要等待它自然结束
                    self.streams[stream_id].join(timeout=1)
                else:
                    self.streams[stream_id].terminate()
                    self.streams[stream_id].wait(timeout=5)

                del self.streams[stream_id]
                del self.stream_configs[stream_id]
                return True
            except Exception as e:
                print(f"Error stopping stream {stream_id}: {str(e)}")
                return False

    def _build_ffmpeg_command(self, config: StreamConfig) -> List[str]:
        base_command = [self.ffmpeg_path, '-re']

        # 输入设置
        if config.stream_type in ['local_video', 'remote_video']:
            if config.loop:
                base_command.extend(['-stream_loop', '-1'])
            base_command.extend(['-i', config.input_source])
        elif config.stream_type == 'rtmp':
            base_command.extend(['-i', config.input_source])

        # 输出设置
        base_command.extend([
            '-c:v', 'libx264',
            '-preset', 'veryfast',
            '-c:a', 'aac',
            '-f', 'flv',
            config.output_url
        ])

        return base_command

    def _push_screen(self, config: StreamConfig):
        sct = mss.mss()
        monitor = sct.monitors[1]  # 主显示器

        ffmpeg_command = [
            self.ffmpeg_path,
            '-f', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-s', f"{monitor['width']}x{monitor['height']}",
            '-r', '30',
            '-i', '-',
            '-c:v', 'libx264',
            '-preset', 'veryfast',
            '-f', 'flv',
            config.output_url
        ]

        process = subprocess.Popen(
            ffmpeg_command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        try:
            while config.stream_id in self.streams:
                screenshot = sct.grab(monitor)
                frame = np.array(screenshot)
                process.stdin.write(frame.tobytes())
        except Exception as e:
            print(f"Error in screen capture: {str(e)}")
        finally:
            if process:
                process.terminate()
                process.wait(timeout=5)

    def get_active_streams(self) -> List[str]:
        with self._lock:
            return list(self.streams.keys())

    def load_stream_list(self, config_file: str) -> bool:
        try:
            with open(config_file, 'r') as f:
                config_data = json.load(f)

            # 检查全局启用状态
            if not config_data.get("enabled", True):
                print("Stream configuration is globally disabled")
                return False

            # 遍历所有流配置
            for stream_config in config_data.get("streams", []):
                # 检查单个流的启用状态
                if not stream_config.get("enabled", True):
                    print(f"Stream {stream_config.get('stream_id')} is disabled, skipping...")
                    continue

                # 移除enabled字段，因为StreamConfig不需要它
                if "enabled" in stream_config:
                    del stream_config["enabled"]

                stream_config_obj = StreamConfig(**stream_config)
                self.start_stream(stream_config_obj)
            return True
        except Exception as e:
            print(f"Error loading stream list: {str(e)}")
            return False


# 使用示例
if __name__ == "__main__":
    pusher = StreamPusher()

    # 从配置文件加载并启动流
    if pusher.load_stream_list("config.json"):
        print("Successfully started streams from config file")

        try:
            # 保持程序运行，直到用户按Ctrl+C
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nStopping all streams...")
            # 停止所有活动的流
            for stream_id in pusher.get_active_streams():
                if pusher.stop_stream(stream_id):
                    print(f"Successfully stopped stream: {stream_id}")
                else:
                    print(f"Failed to stop stream: {stream_id}")
            print("All streams stopped")