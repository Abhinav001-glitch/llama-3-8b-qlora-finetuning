# Llama 3 8B Fine-Tuning with QLoRA

## Overview
This project explores supervised fine-tuning of Meta's Llama 3 8B language model using Unsloth, 4-bit quantization, and LoRA adapters.

The goal is to adapt a pretrained large language model to instruction-following tasks while reducing GPU memory requirements.

## Technologies
- Python
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- TRL (SFTTrainer)
- Unsloth
- PEFT / LoRA
- 4-bit quantization

## Dataset
The project uses the Alpaca Cleaned instruction-following dataset from Hugging Face:

`yahma/alpaca-cleaned`

The dataset contains 51,760 training examples in the recorded run.

## Model and Training
- Base model: `unsloth/llama-3-8b-bnb-4bit`
- Fine-tuning method: LoRA-based supervised fine-tuning
- LoRA rank: 16
- Maximum sequence length: 1,024 tokens
- Learning rate: 0.0002
- Recorded training run: 60 steps
- Trainable parameters: approximately 0.49% of the model

## Hardware
The recorded environment reported two NVIDIA Tesla T4 GPUs. The training output reported using one GPU for the run.

## Repository Contents
- `fine-tuning-llama-3-8b.ipynb`: Training notebook

## How to Run
1. Open the notebook in Google Colab or a compatible GPU environment.
2. Install the required dependencies.
3. Ensure you have access to the selected base model and comply with its license.
4. Run the notebook cells in order.
5. Evaluate the fine-tuned model before using it for real applications.

## Limitations
The recorded experiment ran for 60 training steps. Comprehensive evaluation, comparison against the base model, and inference testing are future improvements.

## Future Improvements
- Evaluate the base and fine-tuned models on the same prompts.
- Save and reload the LoRA adapter.
- Add training-loss plots and sample outputs.
- Document reproducible dependency versions.

## License
Review the base model and dataset licenses before redistributing model artifacts or derivative work.
