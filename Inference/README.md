# Inference Quick Start

This guide shows how to set up the environment and run inference.

## Inference Quick Start

This guide provides instructions for setting up the environment and running inference scripts located in the [inference](./inference/) folder.

### 1. Environment Setup
- Recommended Python version: **3.12.0** (using other versions may cause dependency issues).

```bash
# Example with Conda
conda create -n react_infer_env python=3.12.0
conda activate react_infer_env
```

### 2. Installation

Install the required dependencies:
```bash
pip install -r requirements.txt
```

### 3. Environment Configuration

#### Environment Configuration

Edit the `.env` file to configure your API keys and tool-related settings:

- **SERPER_KEY_ID**: Get your key from [Serper.dev](https://serper.dev/) for web search and Google Scholar
- **JINA_API_KEYS**: Get your key from [Jina.ai](https://jina.ai/) for web page reading
- **API_KEY/API_BASE**: OpenAI-compatible API for page summarization from [OpenAI](https://platform.openai.com/)
- **Proxy(Optional)**: Configure the proxy settings


### 4. Run the Inference Script
You can modify the settings in `run_demo.sh`. 
- **MODEL_PATH**: Path to your model weights
- **QUESTION**: Run DeepResearch inference on the given QUESTION
- **OUTPUT_FILE**: Directory for saving results

Use the following command to start inference:

```bash
bash run_demo.sh
```

#### Web Demo (Optional)

If you prefer a web-based interface for visualizing and interacting with DeepResearch, you can launch the web demo with a single command:

```bash
bash run_web_demo.sh
```

### Acknowledgements

The tool-call in this project is partially inspired by the **[Tongyi DeepResearch](https://github.com/Alibaba-NLP/DeepResearch/tree/main)** work.