#!/usr/bin/env python3

import sherpa_onnx
import sounddevice as sd
import numpy as np
from config import ConfigLoader
cfg = ConfigLoader().config


class KeywordDetector:
    def __init__(self):
        self.keyword_spotter = self._create_keyword_spotter()
        self.sample_rate = 16000  # 唤醒词模型通常使用16kHz采样率
        self.samples_per_read = int(0.1 * self.sample_rate)  # 0.1 second = 100 ms
        self.stream = None
        self.is_awake = False
        
    def _create_keyword_spotter(self):
        """创建唤醒词检测器"""
        # 创建唤醒词检测器，使用 sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01 模型

        try:
            # 确保所有必要的文件存在
            from pathlib import Path
            for file_path in [cfg.kws.tokens, cfg.kws.encoder, cfg.kws.decoder, cfg.kws.joiner, cfg.kws.keywords_file]:
                assert Path(file_path).is_file(), f"{file_path} 文件不存在！"
            
            keyword_spotter = sherpa_onnx.KeywordSpotter(
                tokens=cfg.kws.tokens,
                encoder=cfg.kws.encoder,
                decoder=cfg.kws.decoder,
                joiner=cfg.kws.joiner,
                num_threads=cfg.kws.num_threads if hasattr(cfg.kws, 'num_threads') else 1,
                keywords_file=cfg.kws.keywords_file,
                keywords_score=cfg.kws.keywords_score,
                keywords_threshold=cfg.kws.keywords_threshold,
                num_trailing_blanks=cfg.kws.num_trailing_blanks,
                max_active_paths=cfg.kws.max_active_paths if hasattr(cfg.kws, 'max_active_paths') else 4,
                provider=cfg.kws.provider if hasattr(cfg.kws, 'provider') else "cpu",
            )
            return keyword_spotter
        except Exception as e:
            print(f"初始化唤醒词检测器失败: {e}")
            return None
    
    def create_stream(self):
        """创建唤醒词检测流"""
        if self.keyword_spotter is None:
            print("唤醒词检测器未初始化，无法创建流")
            return None
        
        try:
            self.stream = self.keyword_spotter.create_stream()
            return self.stream
        except Exception as e:
            print(f"创建唤醒词检测流失败: {e}")
            return None
    
    def reset_stream(self):
        """重置唤醒词检测流"""
        if self.stream is not None and self.keyword_spotter is not None:
            try:
                self.keyword_spotter.reset_stream(self.stream)
                return True
            except Exception as e:
                print(f"重置唤醒词检测流失败: {e}")
                return False
        return False
    
    def process_audio(self, samples):
        if self.keyword_spotter is None or self.stream is None:
            return False
        
        # 接受音频数据
        self.stream.accept_waveform(self.sample_rate, samples)
        
        # 检测唤醒词
        detected = False
        while self.keyword_spotter.is_ready(self.stream):
            self.keyword_spotter.decode_stream(self.stream)
            result = self.keyword_spotter.get_result(self.stream)
            if result:
                print(f"检测到唤醒词: {result}")
                detected = True
                self.is_awake = True
                # 重置流，准备下一次检测
                self.keyword_spotter.reset_stream(self.stream)
                break
        
        return detected
    
    def check_devices(self):
        """检查并显示音频设备"""
        devices = sd.query_devices()
        if len(devices) == 0:
            print("No microphone devices found")
            return False
        
        print(devices)
        default_input_device_idx = sd.default.device[0]
        print(f'Use default device: {devices[default_input_device_idx]["name"]}')
        return True

    def start_listening(self):
        """开始监听唤醒词"""
        if not self.check_devices():
            return False
        
        if self.keyword_spotter is None:
            print("唤醒词检测器未初始化，无法开始监听")
            return False
        
        self.create_stream()
        print("开始监听唤醒词...")
        self.is_awake = False
        
        try:
            with sd.InputStream(channels=1, dtype="float32", samplerate=self.sample_rate) as s:
                while not self.is_awake:
                    samples, _ = s.read(self.samples_per_read)
                    samples = samples.reshape(-1)
                    if self.process_audio(samples):
                        break
            
            return True
        except Exception as e:
            print(f"监听唤醒词时发生错误: {e}")
            return False

# 如果直接运行此文件，进行测试
if __name__ == "__main__":
    detector = KeywordDetector()
    try:
        while True:
            print("等待唤醒...")
            if detector.start_listening():
                print("已唤醒！执行后续操作...")
                # 这里可以添加唤醒后的操作
                detector.is_awake = False  # 重置唤醒状态，准备下一次唤醒
    except KeyboardInterrupt:
        print("\nCaught Ctrl + C. Exiting")