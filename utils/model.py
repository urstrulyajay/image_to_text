from unsloth import FastVisionModel
import torch

def load_model(config):
    model, tokenizer = FastVisionModel.from_pretrained(
        config["model_name"],
        load_in_4bit=config["load_in_4bit"],
        use_gradient_checkpointing=config["use_gradient_checkpointing"],
    )

    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=config["finetune_vision_layers"],
        finetune_language_layers=config["finetune_language_layers"],
        finetune_attention_modules=config["finetune_attention_modules"],
        finetune_mlp_modules=config["finetune_mlp_modules"],
        r=config["r"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        bias=config["bias"],
        random_state=config["random_state"],
        use_rslora=config["use_rslora"],
        loftq_config=config["loftq_config"],
    )

    return model, tokenizer
