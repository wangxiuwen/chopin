import asyncio
import io
import edge_tts
from pydub import AudioSegment
from pydub.playback import play

class TextToSpeechPlayer:
    def __init__(self, voice: str = "zh-CN-YunxiNeural", rate: str = "+0%", volume: str = "+0%"):
        """
        初始化语音播放器
        :param voice: 语音名称，默认使用中文语音"zh-CN-YunxiNeural"
        :param rate: 语速调整，默认+0%
        :param volume: 音量调整，默认+0%
        """
        self.voice = voice
        self.rate = rate
        self.volume = volume

    async def play_text(self, text: str) -> None:
        """
        异步播放文本
        :param text: 要转换为语音的文本
        """
        # 创建Edge TTS通信对象
        communicate = edge_tts.Communicate(
            text, 
            self.voice, 
            rate=self.rate, 
            volume=self.volume
        )
        
        # 收集音频数据到内存
        audio_stream = io.BytesIO()
        
        # 流式接收音频数据
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_stream.write(chunk["data"])
        
        # 重置指针并加载音频
        audio_stream.seek(0)
        audio = AudioSegment.from_file(audio_stream, format="mp3")
        
        # 在异步环境中播放音频
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, play, audio)

if __name__ == "__main__":
    # 测试代码
    async def main():
        import sys
        # 获取用户输入或使用默认文本
        text = sys.argv[1] if len(sys.argv) > 1 else "欢迎使用edge-TTS语音合成服务。"
        
        # 创建播放器实例
        tts_player = TextToSpeechPlayer()
        
        # 播放文本
        print("正在生成并播放语音...")
        await tts_player.play_text(text)
        print("播放完成！")
    
    # 运行主函数
    asyncio.run(main())