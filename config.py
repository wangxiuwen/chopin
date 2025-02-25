import yaml
from pathlib import Path
from types import SimpleNamespace

class ConfigLoader:
    def __init__(self, config_path="config.yaml"):
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        
        # 将字典转换为嵌套的简单对象
        self.config = SimpleNamespace()
        for section, values in config_dict.items():
            setattr(self.config, section, SimpleNamespace(**values))

    def __getattr__(self, name):
        return getattr(self.config, name)

# 使用示例
if __name__ == "__main__":
    config = ConfigLoader().config
    print(config.audio.sampling_rate)  # 输出: 16000
    print(config.llm.base_url)         # 输出: https://dashscope.aliyuncs.com/compatible-mode/v1