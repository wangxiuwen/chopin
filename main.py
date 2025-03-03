#!/usr/bin/env python3

import asyncio
from asr import SpeechRecognizer
from llm import LLM
import sounddevice as sd
from tts import TextToSpeechPlayer

async def main():
    # 初始化语音识别器和LLM
    recognizer = SpeechRecognizer()
    llm = LLM()
    tts_player = TextToSpeechPlayer(speed=1.0)

    # 系统提示消息
    system_message = {
        'role': 'system',
        'content': 'You are a helpful assistant.'
    }
    
    print("Started! Please speak")
    last_result = ""
    segment_id = 0
    
    # 检查音频设备
    if not recognizer.check_devices():
        return

    try:
        with sd.InputStream(channels=1, dtype="float32", samplerate=recognizer.sample_rate) as s:
            while True:
                # 读取语音输入
                samples, _ = s.read(recognizer.samples_per_read)
                samples = samples.reshape(-1)
                recognizer.stream.accept_waveform(recognizer.sample_rate, samples)
                
                while recognizer.recognizer.is_ready(recognizer.stream):
                    recognizer.recognizer.decode_stream(recognizer.stream)

                is_endpoint = recognizer.recognizer.is_endpoint(recognizer.stream)
                result = recognizer.recognizer.get_result(recognizer.stream)

                # 显示实时识别结果
                if result and (last_result != result):
                    last_result = result
                    print("\rSpeech recognized: {}:{}".format(segment_id, result), end="", flush=True)
                
                # 当检测到语音端点时处理
                if is_endpoint:
                    if result:
                        print("\nFinal result {}:{}".format(segment_id, result))
                        
                        # 创建消息列表并调用 LLM
                        messages = [
                            system_message,
                            {'role': 'user', 'content': result}
                        ]
                        llm_response = llm.call(messages)
                        print(f"LLM response: {llm_response}\n")
                        await tts_player.play_text(llm_response)  # 正确使用 await
                        print("Audio playback completed!")
                        segment_id += 1
                    recognizer.recognizer.reset(recognizer.stream)
                    
    except KeyboardInterrupt:
        print("\nCaught Ctrl + C. Exiting")

if __name__ == "__main__":
    asyncio.run(main())  # 使用 asyncio.run 运行异步函数