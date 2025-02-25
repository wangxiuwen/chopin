#!/usr/bin/env python3

import sounddevice as sd
import sherpa_onnx
from config import ConfigLoader

class SpeechRecognizer:
    def __init__(self):
        self.cfg = ConfigLoader().config
        self.recognizer = self._create_recognizer()
        self.sample_rate = 48000  # 可以作为参数传入，这里保留默认值
        self.samples_per_read = int(0.1 * self.sample_rate)  # 0.1 second = 100 ms
        self.stream = self.recognizer.create_stream()
        
    def _create_recognizer(self):
        """创建语音识别器"""
        recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=self.cfg.asr.tokens,
            encoder=self.cfg.asr.encoder,
            decoder=self.cfg.asr.decoder,
            joiner=self.cfg.asr.joiner,
            num_threads=1,
            sample_rate=16000,
            feature_dim=80,
            enable_endpoint_detection=True,
            rule1_min_trailing_silence=2.4,
            rule2_min_trailing_silence=1.2,
            rule3_min_utterance_length=300,
        )
        return recognizer

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

    def start_recognition(self):
        """开始语音识别"""
        if not self.check_devices():
            return

        print("Started! Please speak")
        last_result = ""
        segment_id = 0
        
        try:
            with sd.InputStream(channels=1, dtype="float32", samplerate=self.sample_rate) as s:
                while True:
                    samples, _ = s.read(self.samples_per_read)
                    samples = samples.reshape(-1)
                    self.stream.accept_waveform(self.sample_rate, samples)
                    
                    while self.recognizer.is_ready(self.stream):
                        self.recognizer.decode_stream(self.stream)

                    is_endpoint = self.recognizer.is_endpoint(self.stream)
                    result = self.recognizer.get_result(self.stream)

                    if result and (last_result != result):
                        last_result = result
                        print("\r last result-------哈哈 {}:{}".format(segment_id, result), end="", flush=True)
                    
                    if is_endpoint:
                        if result:
                            print("\r{}:{}".format(segment_id, result), flush=True)
                            segment_id += 1
                        self.recognizer.reset(self.stream)
                    
        except KeyboardInterrupt:
            print("\nCaught Ctrl + C. Exiting")

# 如果直接运行此文件，仍可测试
if __name__ == "__main__":
    recognizer = SpeechRecognizer()
    recognizer.start_recognition()