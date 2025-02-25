# Gemma-2 Fine-Tuning on Custom Dataset

## Overview
This project demonstrates the process of fine-tuning the Gemma-2 language model using a custom dataset. The fine-tuning approach utilizes Low-Rank Adaptation (LoRa) to enhance the model's performance with minimal additional parameters. The results before and after fine-tuning showcase notable improvements in the model's responses.

## Dataset
The dataset used for fine-tuning is the Databricks Dolly 15K, sourced from Hugging Face. It is provided in JSON Lines (JSONL) format, and 1,000 samples were selected for training.

## Environment Setup
Install the necessary libraries:
```bash
pip install numpy pandas keras keras-nlp keras-hub
```
Set up the environment for optimal GPU usage:
```python
import os
os.environ["KERAS_BACKEND"] = "jax"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1.00"
```

## Loading the Dataset
```python
import json
with open("databricks-dolly-15k.jsonl") as file:
    data = [json.loads(line) for line in file if json.loads(line).get("context") is None]
data = [f"Instruction:\n{item['instruction']}\n\nResponse:\n{item['response']}" for item in data[:1000]]
```

## Model Initialization
```python
import keras_hub
gemma_lm = keras_hub.models.GemmaCausalLM.from_preset("gemma2_2b_en")
gemma_lm.summary()
```

## Fine-Tuning with LoRa
```python
gemma_lm.backbone.enable_lora(rank=4)
gemma_lm.preprocessor.sequence_length = 256
optimizer = keras.optimizers.AdamW(learning_rate=5e-5, weight_decay=0.01)
optimizer.exclude_from_weight_decay(var_names=["bias", "scale"])
gemma_lm.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                 optimizer=optimizer,
                 weighted_metrics=[keras.metrics.SparseCategoricalAccuracy()])
gemma_lm.fit(data, epochs=5, batch_size=1)
```

## Results
Before fine-tuning, the model's responses were more generalized and less specific. After fine-tuning, the outputs became more detailed, contextually accurate, and aligned with the given instructions.

## Optimization Tips
- Increase the dataset size for better results.
- Train for more epochs to enhance performance.
- Use a higher LoRa rank for more expressive outputs.
- Adjust learning rate and weight decay for optimal convergence.

## References
- The Databricks Dolly 15K dataset is available on Hugging Face.
- Keras NLP documentation provides comprehensive guidance on using Keras for NLP tasks.

## References
- [The Databricks Dolly 15K dataset](https://huggingface.co/datasets/databricks/databricks-dolly-15k) is available on Hugging Face.
- [Keras Hub documentation](https://keras.io/keras_hub/) provides comprehensive guidance on using Keras for NLP tasks.

## License
This project is licensed under the Apache License.

