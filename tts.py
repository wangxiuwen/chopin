import asyncio
import io
import sherpa_onnx
import numpy as np
import soundfile as sf
from pydub import AudioSegment
from pydub.playback import play
from config import ConfigLoader

class TextToSpeechPlayer:
    def __init__(self, voice: str = "zh-CN-YunxiNeural", rate: str = "+0%", volume: str = "+0%"):
        """
        初始化语音播放器
        :param voice: 语音名称，在sherpa-onnx中不使用，保留参数以兼容现有代码
        :param rate: 语速调整，对应sherpa-onnx中的speed参数，默认1.0
        :param volume: 音量调整，在sherpa-onnx中不使用，保留参数以兼容现有代码
        """
        # 保留原始参数以兼容现有代码
        self.voice = voice
        self.rate = rate
        self.volume = volume
        
        # 解析rate参数，将edge-tts的格式(如"+0%")转换为sherpa-onnx的speed参数(如1.0)
        try:
            # 去掉百分号并转换为浮点数
            rate_value = float(self.rate.rstrip('%').replace('+', ''))
            # 转换为sherpa-onnx的speed参数格式：+10%对应1.1，-10%对应0.9
            self.speed = 1.0 + (rate_value / 100.0)
        except ValueError:
            # 如果解析失败，使用默认值1.0
            self.speed = 1.0
            
        # 初始化TTS配置
        self._init_tts()

    def _init_tts(self):
        """
        初始化sherpa-onnx TTS配置
        """
        try:
            # 从配置文件加载TTS模型路径
            cfg = ConfigLoader().config
            # 如果配置中有tts部分，则使用配置中的路径
            if hasattr(cfg, 'tts'):
                model_path = getattr(cfg.tts, 'vits-model', getattr(cfg.tts, 'model', None))
                lexicon_path = getattr(cfg.tts, 'vits-lexicon', getattr(cfg.tts, 'lexicon', None))
                tokens_path = getattr(cfg.tts, 'vits-tokens', getattr(cfg.tts, 'tokens', None))
            else:
                # 使用默认路径 - 根据可用模型选择
                model_path = "./models/vits-icefall-zh-aishell3/model.onnx"
                lexicon_path = "./models/vits-icefall-zh-aishell3/lexicon.txt"
                tokens_path = "./models/vits-icefall-zh-aishell3/tokens.txt"
            
            # 创建VITS模型配置
            vits_config = sherpa_onnx.OfflineTtsVitsModelConfig(
                model=model_path,
                lexicon=lexicon_path,
                tokens=tokens_path,
                dict_dir=getattr(cfg.tts, 'vits-dict-dir', None)
            )
            
            # 创建TTS模型配置
            model_config = sherpa_onnx.OfflineTtsModelConfig(
                vits=vits_config,
                num_threads=1
            )
            
            # 创建TTS配置
            config = sherpa_onnx.OfflineTtsConfig(
                model=model_config
            )
            
            # 创建TTS实例
            self.tts = sherpa_onnx.OfflineTts(config)
            self.sid = 0  # 默认使用说话人ID 0
            
        except Exception as e:
            print(f"初始化TTS失败: {e}")
            self.tts = None
    
    async def play_text(self, text: str) -> None:
        """
        异步播放文本
        :param text: 要转换为语音的文本
        """
        if self.tts is None:
            print("TTS未初始化，无法播放语音")
            return
        
        try:
            # 在异步环境中生成语音
            loop = asyncio.get_event_loop()
            audio = await loop.run_in_executor(
                None, 
                lambda: self.tts.generate(text, sid=self.sid, speed=self.speed)
            )
            
            # 获取音频数据和采样率
            samples = audio.samples
            sample_rate = audio.sample_rate
            
            # 将numpy数组转换为AudioSegment并播放
            audio_bytes = io.BytesIO()
            await loop.run_in_executor(
                None,
                lambda: sf.write(audio_bytes, samples, sample_rate, format='wav')
            )
            audio_bytes.seek(0)
            
            # 加载为AudioSegment并播放
            audio_segment = await loop.run_in_executor(
                None,
                lambda: AudioSegment.from_file(audio_bytes, format="wav")
            )
            
            # 播放音频
            await loop.run_in_executor(None, play, audio_segment)
            
        except Exception as e:
            print(f"生成或播放语音失败: {e}")

if __name__ == "__main__":
    # 测试代码
    async def main():
        import sys
        # 获取用户输入或使用默认文本
        text = sys.argv[1] if len(sys.argv) > 1 else "欢迎使用sherpa-onnx语音合成服务。"
        
        # 创建播放器实例
        tts_player = TextToSpeechPlayer()
        
        # 播放文本
        print("正在生成并播放语音...")
        await tts_player.play_text(text)
        print("播放完成！")
    
    # 运行主函数
    asyncio.run(main())