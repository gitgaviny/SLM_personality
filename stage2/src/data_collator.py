# Create by Hao at 2025-07-01
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
import torch


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor ([`WhisperProcessor`])
            The processor used for processing the data.
        decoder_start_token_id (`int`)
            The begin-of-sentence of the decoder.
        forward_attention_mask (`bool`)
            Whether to return attention_mask.
    """

    processor: Any
    decoder_start_token_id: int
    decoder_end_token_id: int
    config: Any

    # filled in __post_init__
    forward_attention_mask: bool = field(init=False)

    def __post_init__(self):
        self.forward_attention_mask = (
            getattr(self.config, "model_type", None) == "whisper"
            and getattr(self.config, "apply_spec_augment", False)
            and getattr(self.config, "mask_time_prob", 0) > 0
        )

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need
        # different padding methods
        model_input_name = self.processor.model_input_names[0]
        input_features = [{model_input_name: feature[model_input_name]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        if self.forward_attention_mask:
            batch["attention_mask"] = torch.LongTensor([feature["attention_mask"] for feature in features])

        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), self.config.ignore_token_id)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        # We DO NOT insert the <eos> token here since the current label is also treated as the input
        # If we insert it here, we should delete it in the later stage
        # So we directly insert it in the later stage
        batch["labels"] = labels
        # The self.processor.tokenizer.pad should use "input_ids" as input feature
        # Thus, we put the feature["prompt_ids"] as "input_ids"
        prompts_features= [{"input_ids": feature["prompt_ids"]} for feature in features]
        prompts_batch = self.processor.tokenizer.pad(prompts_features, return_tensors="pt")
        batch["prompt_ids"] = prompts_batch["input_ids"]

        return batch

