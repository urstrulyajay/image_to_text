import json
from model import load_model
from dataset import ImageTextJSONDataset, convert_to_conversation
from torch.utils.data import DataLoader
from unsloth import is_bf16_supported
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig

with open("config.json") as f:
    config = json.load(f)

model, tokenizer = load_model(config)

FastVisionModel = config["FastVisionModel"]
FastVisionModel.for_training(model)

dataset = ImageTextJSONDataset(config["dataset_path"], config["image_size"])
dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

converted_dataset = [convert_to_conversation(sample, config["instruction"]) for sample in dataset]

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    data_collator = UnslothVisionDataCollator(model, tokenizer),
    train_dataset = converted_dataset,
    args = SFTConfig(
        per_device_train_batch_size = config["train_batch_size"],
        gradient_accumulation_steps = config["gradient_accumulation_steps"],
        warmup_steps = config["warmup_steps"],
        max_steps = config["max_steps"],
        learning_rate = config["learning_rate"],
        fp16 = not is_bf16_supported(),
        bf16 = is_bf16_supported(),
        logging_steps = config["logging_steps"],
        optim = config["optim"],
        weight_decay = config["weight_decay"],
        lr_scheduler_type = config["lr_scheduler_type"],
        seed = config["seed"],
        output_dir = config["output_dir"],
        report_to = config["report_to"],
        remove_unused_columns = config["remove_unused_columns"],
        dataset_text_field = config["dataset_text_field"],
        dataset_kwargs = config["dataset_kwargs"],
        dataset_num_proc = config["dataset_num_proc"],
        max_seq_length = config["max_seq_length"],
    ),
)

trainer_stats = trainer.train()
model.save_pretrained(config["save_dir"])
tokenizer.save_pretrained(config["save_dir"])