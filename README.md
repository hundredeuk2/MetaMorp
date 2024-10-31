## Introduction
Like-minded projects that use the Vllm framework to quickly infer data from models of your choice from anywhere!

## Usage
### Poetry Initialize
after install poetry in your server

```bash
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

```bash
cd your/project/metamorp
poetry env use python
poetry install
poetry shell
```

### Quickstart
```bash
python inference.py --config_path configs/base.yaml
```
or

```bash
sh run.sh
```

### Data setting
config 파일에 prompt중 system_prompt와 input_prompt에서 사용하고싶은 데이터가 포함되야하는 경우 데이터의 columns를 프롬프트에 적용하거나, 프롬프트 {}안에 있는 변수명으로 데이터 columns명을 맞춰주면 됩니다.
! 아직은 data 저장 type이 csv만 됩니다!!!