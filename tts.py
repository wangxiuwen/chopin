import asyncio
import io
import sherpa_onnx
import numpy as np
import soundfile as sf
from pydub import AudioSegment
from pydub.playback import play
from config import ConfigLoader  # 导入配置加载器

cfg = ConfigLoader().config

class TextToSpeechPlayer:
    def __init__(self, speed=1.0):
        # 初始化TTS配置
        self.speed = speed  # 保存speed参数为实例变量
        self._init_tts()

    def _init_tts(self):

        # 创建VITS模型配置
        vits_config = sherpa_onnx.OfflineTtsVitsModelConfig(
            model=cfg.tts.model,
            lexicon=cfg.tts.lexicon,
            tokens=cfg.tts.tokens
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
    
    def _preprocess_text(self, text: str) -> str:
        """
        预处理文本，移除或替换可能导致OOV错误的特殊字符
        :param text: 原始文本
        :return: 处理后的文本
        """
        # 定义需要替换的特殊字符及其替换字符
        replacements = {
            '）': ')',  # 全角括号替换为半角
            '（': '(',
            '。': '.',  # 全角标点替换为半角
            '，': ',',
            '！': '!',
            '？': '?',
            '［': '[',  # 全角方括号替换为半角
            '］': ']',
            '【': '[',  # 全角方括号（另一种）替换为半角
            '】': ']'
        }
        
        # 应用替换规则
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        return text

    async def play_text(self, text: str):
        """生成并播放语音"""
        try:
            # 预处理文本
            text = self._preprocess_text(text)
            
            # 生成语音
            audio = self.tts.generate(text, sid=self.sid, speed=self.speed)
            
            # 获取音频数据和采样率
            samples = audio.samples
            sample_rate = audio.sample_rate
            
            # 获取当前事件循环
            loop = asyncio.get_running_loop()
            
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
        tts_player = TextToSpeechPlayer()  # 提供默认的speed参数
        
        # 播放文本
        print("正在生成并播放语音...")
        await tts_player.play_text(text)
        print("播放完成！")
    
    # 运行主函数
    asyncio.run(main())