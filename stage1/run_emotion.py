import logging
import pathlib
import re
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Union
import evaluate
import datasets
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging import version
import torch.distributed as dist
import librosa
from lang_trans import arabic

import soundfile as sf
from model import Wav2Vec2ForCTCnCLS
from transformers.trainer_utils import get_last_checkpoint
import os

from transformers import (
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
    is_apex_available,
    trainer_utils,
)

if is_apex_available():
    from apex import amp

if version.parse(torch.__version__) >= version.parse("1.6"):
    _is_native_amp_available = True
    from torch.cuda.amp import autocast


logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    freeze_feature_extractor: Optional[bool] = field(
        default=False, metadata={"help": "Whether to freeze the feature extractor layers of the model."}
    )
    verbose_logging: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to log verbose messages or not."},
    )
    alpha: Optional[float] = field(
        default=0.1,
        metadata={"help": "loss_cls + alpha * loss_ctc"},
    )
    tokenizer: Optional[str] = field(
        default=None,
        metadata={"help": "Path to pretrained tokenizer"}
    )


def configure_logger(model_args: ModelArguments, training_args: TrainingArguments):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logging_level = logging.WARNING
    if model_args.verbose_logging:
        logging_level = logging.DEBUG
    elif trainer_utils.is_main_process(training_args.local_rank):
        logging_level = logging.INFO
    logger.setLevel(logging_level)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using HfArgumentParser we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    dataset_name: str = field(
        default='emotion', metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_split_name: Optional[str] = field(
        default="train",
        metadata={
            "help": "The name of the training data set split to use (via the datasets library). Defaults to 'train'"
        },
    )
    validation_split_name: Optional[str] = field(
        default="validation",
        metadata={
            "help": "The name of the validation data set split to use (via the datasets library). Defaults to 'validation'"
        },
    )
    target_text_column: Optional[str] = field(
        default="text",
        metadata={"help": "Column in the dataset that contains label (target text). Defaults to 'text'"},
    )
    speech_file_column: Optional[str] = field(
        default="file",
        metadata={"help": "Column in the dataset that contains speech file path. Defaults to 'file'"},
    )
    target_feature_extractor_sampling_rate: Optional[bool] = field(
        default=False,
        metadata={"help": "Resample loaded audio to target feature extractor's sampling rate or not."},
    )
    max_duration_in_seconds: Optional[float] = field(
        default=None,
        metadata={"help": "Filters out examples longer than specified. Defaults to no filtering."},
    )
    orthography: Optional[str] = field(
        default="librispeech",
        metadata={
            "help": "Orthography used for normalization and tokenization: 'librispeech' (default), 'timit', or 'buckwalter'."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )

    # select which split as test
    split_id: str = field(
        default='01F', metadata={"help": "iemocap_ + splitid (e.g. 01M, 02F, etc) + train/test.csv"}
    )

    output_file: Optional[str] = field(
        default=None,
        metadata={"help": "Output file."},
    )


@dataclass
class Orthography:
    """
    Orthography scheme used for text normalization and tokenization.

    Args:
        do_lower_case (:obj:bool, optional, defaults to :obj:False):
            Whether or not to accept lowercase input and lowercase the output when decoding.
        vocab_file (:obj:str, optional, defaults to :obj:None):
            File containing the vocabulary.
        word_delimiter_token (:obj:str, optional, defaults to :obj:"|"):
            The token used for delimiting words; it needs to be in the vocabulary.
        translation_table (:obj:Dict[str, str], optional, defaults to :obj:{}):
            Table to use with str.translate() when preprocessing text (e.g., "-" -> " ").
        words_to_remove (:obj:Set[str], optional, defaults to :obj:set()):
            Words to remove when preprocessing text (e.g., "sil").
        untransliterator (:obj:Callable[[str], str], optional, defaults to :obj:None):
            Function that untransliterates text back into native writing system.
    """

    do_lower_case: bool = False
    vocab_file: Optional[str] = None
    word_delimiter_token: Optional[str] = "|"
    translation_table: Optional[Dict[str, str]] = field(default_factory=dict)
    words_to_remove: Optional[Set[str]] = field(default_factory=set)
    untransliterator: Optional[Callable[[str], str]] = None
    tokenizer: Optional[str] = None

    @classmethod
    def from_name(cls, name: str):
        if name == "librispeech":
            return cls()
        if name == "timit":
            return cls(
                do_lower_case=True,
                # break compounds like "quarter-century-old" and replace pauses "--"
                translation_table=str.maketrans({"-": " "}),
            )
        if name == "buckwalter":
            translation_table = {
                "-": " ",  # sometimes used to represent pauses
                "^": "v",  # fixing "tha" in arabic_speech_corpus dataset
            }
            return cls(
                vocab_file=pathlib.Path(__file__).parent.joinpath("vocab/buckwalter.json"),
                word_delimiter_token="/",  # "|" is Arabic letter alef with madda above
                translation_table=str.maketrans(translation_table),
                words_to_remove={"sil"},  # fixing "sil" in arabic_speech_corpus dataset
                untransliterator=arabic.buckwalter.untransliterate,
            )
        raise ValueError(f"Unsupported orthography: '{name}'.")

    def preprocess_for_training(self, text: str) -> str:
        # TODO(elgeish) return a pipeline (e.g., from jiwer) instead? Or rely on branch predictor as is
        if len(self.translation_table) > 0:
            text = text.translate(self.translation_table)
        if len(self.words_to_remove) == 0:
            try:
                text = " ".join(text.split())  # clean up whitespaces
            except:
                text = "NULL"
        else:
            text = " ".join(w for w in text.split() if w not in self.words_to_remove)  # and clean up whilespaces
        return text

    def create_processor(self, model_args: ModelArguments) -> Wav2Vec2Processor:
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            model_args.model_name_or_path, cache_dir=model_args.cache_dir
        )
        if self.vocab_file:
            tokenizer = Wav2Vec2CTCTokenizer(
                self.vocab_file,
                cache_dir=model_args.cache_dir,
                do_lower_case=self.do_lower_case,
                word_delimiter_token=self.word_delimiter_token,
            )
        else:
            tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(
                self.tokenizer,
                cache_dir=model_args.cache_dir,
                do_lower_case=self.do_lower_case,
                word_delimiter_token=self.word_delimiter_token,
            )
        return Wav2Vec2Processor(feature_extractor, tokenizer)


@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:~transformers.Wav2Vec2Processor)
            The processor used for proccessing the data.
        padding (:obj:bool, :obj:str or :class:~transformers.tokenization_utils_base.PaddingStrategy, optional, defaults to :obj:True):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:True or :obj:'longest': Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:'max_length': Pad to a maximum length specified with the argument :obj:max_length or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:False or :obj:'do_not_pad' (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:int, optional):
            Maximum length of the `input_values of the returned list and optionally padding length (see above).
        max_length_labels (:obj:int, optional):
            Maximum length of the `labels returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:int, optional):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None
    audio_only = False

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        if self.audio_only is False:
            label_features = [{"input_ids": feature["labels"][:-1]} for feature in features]
            cls_labels = [feature["labels"][-1] for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        if self.audio_only is False:
            with self.processor.as_target_processor():
                labels_batch = self.processor.pad(
                    label_features,
                    padding=self.padding,
                    max_length=self.max_length_labels,
                    pad_to_multiple_of=self.pad_to_multiple_of_labels,
                    return_tensors="pt",
                )

            # replace padding with -100 to ignore loss correctly
            ctc_labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
            batch["labels"] = (ctc_labels, torch.tensor(cls_labels))  # labels = (ctc_labels, cls_labels)

        return batch


class CTCTrainer(Trainer):
    def _prepare_inputs(self, inputs: Dict[str, Union[torch.Tensor, Any]]) -> Dict[str, Union[torch.Tensor, Any]]:
        # 将所有 tensor 或 labels 中的张量迁移到正确设备（兼容 deepspeed 的 dtype）
        for k, v in list(inputs.items()):
            if isinstance(v, torch.Tensor):
                kwargs = dict(device=self.args.device)
                if getattr(self, "deepspeed", None) and v.dtype != torch.int64:
                    kwargs.update(dict(dtype=self.args.hf_deepspeed_config.dtype()))
                inputs[k] = v.to(**kwargs)

            # labels 是一个 (ctc_labels, cls_labels) 的 tuple，需要逐个搬运
            if k == "labels" and isinstance(v, (tuple, list)):
                moved = []
                for t in v:
                    if isinstance(t, torch.Tensor):
                        kwargs = dict(device=self.args.device)
                        if getattr(self, "deepspeed", None) and t.dtype != torch.int64:
                            kwargs.update(dict(dtype=self.args.hf_deepspeed_config.dtype()))
                        moved.append(t.to(**kwargs))
                    else:
                        moved.append(t)
                inputs[k] = tuple(moved)

        if self.args.past_index >= 0 and self._past is not None:
            inputs["mems"] = self._past

        return inputs

    # 不重写 training_step，使用 Trainer 默认实现（兼容 Accelerate/AMP/ZeRO）


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))

    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    configure_logger(model_args, training_args)

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the --output_dir or add --overwrite_output_dir to train from scratch."
            )

    orthography = Orthography.from_name(data_args.orthography.lower())
    orthography.tokenizer = model_args.tokenizer
    processor = orthography.create_processor(model_args)

    if data_args.dataset_name == 'emotion':
        train_dataset = datasets.load_dataset('csv', data_files='/lustre/users/gao/speechllm/labels/iemocap_labels_all/session' + data_args.split_id + '.train.csv', cache_dir=model_args.cache_dir)['train']
        val_dataset = datasets.load_dataset('csv', data_files='/lustre/users/gao/speechllm/labels/iemocap_labels_all/session' + data_args.split_id + '.test.csv', cache_dir=model_args.cache_dir)['train']
        cls_label_map = {"neutral":0, "positive":1, "negative":2}

    model = Wav2Vec2ForCTCnCLS.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        # gradient_checkpointing=training_args.gradient_checkpointing,
        vocab_size=len(processor.tokenizer),
        cls_len=len(cls_label_map),
        alpha=model_args.alpha,
    )

    wer_metric = evaluate.load("wer")
    target_sr = processor.feature_extractor.sampling_rate if data_args.target_feature_extractor_sampling_rate else None
    vocabulary_chars_str = "".join(t for t in processor.tokenizer.get_vocab().keys() if len(t) == 1)
    vocabulary_text_cleaner = re.compile(  # remove characters not in vocabulary
        f"[^\s{re.escape(vocabulary_chars_str)}]",  # allow space in addition to chars in vocabulary
        flags=re.IGNORECASE if processor.tokenizer.do_lower_case else 0,
    )
    text_updates = []

    def prepare_example(example, audio_only=False):  # TODO(elgeish) make use of multiprocessing?
        example["speech"], example["sampling_rate"] = librosa.load(example[data_args.speech_file_column], sr=target_sr)
        if data_args.max_duration_in_seconds is not None:
            example["duration_in_seconds"] = len(example["speech"]) / example["sampling_rate"]
        if audio_only is False:
            # Normalize and clean up text; order matters!
            updated_text = orthography.preprocess_for_training(example[data_args.target_text_column])
            updated_text = vocabulary_text_cleaner.sub("", updated_text)
            if updated_text != example[data_args.target_text_column]:
                text_updates.append((example[data_args.target_text_column], updated_text))
                example[data_args.target_text_column] = updated_text
        return example

    if training_args.do_train:
        train_dataset = train_dataset.map(prepare_example, remove_columns=[data_args.speech_file_column])
    if training_args.do_predict:
        val_dataset = val_dataset.map(prepare_example, fn_kwargs={'audio_only': False})
    elif training_args.do_eval:
        val_dataset = val_dataset.map(prepare_example, remove_columns=[data_args.speech_file_column])

    if data_args.max_duration_in_seconds is not None:
        def filter_by_max_duration(example):
            return example["duration_in_seconds"] <= data_args.max_duration_in_seconds
        if training_args.do_train:
            old_train_size = len(train_dataset)
            train_dataset = train_dataset.filter(filter_by_max_duration, remove_columns=["duration_in_seconds"])
            if len(train_dataset) > old_train_size:
                logger.warning(
                    f"Filtered out {len(train_dataset) - old_train_size} train example(s) longer than {data_args.max_duration_in_seconds} second(s)."
                )
        if training_args.do_predict or training_args.do_eval:
            old_val_size = len(val_dataset)
            val_dataset = val_dataset.filter(filter_by_max_duration, remove_columns=["duration_in_seconds"])
            if len(val_dataset) > old_val_size:
                logger.warning(
                    f"Filtered out {len(val_dataset) - old_val_size} validation example(s) longer than {data_args.max_duration_in_seconds} second(s)."
                )
    # logger.info(f"Split sizes: {len(train_dataset)} train and {len(val_dataset)} validation.")

    logger.warning(f"Updated {len(text_updates)} transcript(s) using '{data_args.orthography}' orthography rules.")
    if logger.isEnabledFor(logging.DEBUG):
        for original_text, updated_text in text_updates:
            logger.debug(f'Updated text: "{original_text}" -> "{updated_text}"')
    text_updates = None

    def prepare_dataset(batch, audio_only=False):
        # check that all files have the correct sampling rate
        assert (
            len(set(batch["sampling_rate"])) == 1
        ), f"Make sure all inputs have the same sampling rate of {processor.feature_extractor.sampling_rate}."
        batch["input_values"] = processor(batch["speech"], sampling_rate=batch["sampling_rate"][0]).input_values
        if audio_only is False:
            cls_labels = list(map(lambda e: cls_label_map[e], batch["sentiment"]))
            tokenized = processor.tokenizer(batch[data_args.target_text_column])
            batch["labels"] = tokenized["input_ids"]
            for i, y in enumerate(cls_labels):
                batch["labels"][i].append(y)
        return batch

    if training_args.do_train:
        train_dataset = train_dataset.map(
            prepare_dataset,
            batch_size=training_args.per_device_train_batch_size,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
        )
    if training_args.do_predict:
        val_dataset = val_dataset.map(
            prepare_dataset,
            fn_kwargs={'audio_only': False},
            batch_size=training_args.per_device_train_batch_size,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
        )
    elif training_args.do_eval:
        val_dataset = val_dataset.map(
            prepare_dataset,
            batch_size=training_args.per_device_train_batch_size,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
        )

    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

    def compute_metrics(pred):
        cls_pred_logits = pred.predictions[1]
        cls_pred_ids = np.argmax(cls_pred_logits, axis=-1).astype(int)

        y_true = np.asarray(pred.label_ids[1]).astype(int).reshape(-1)
        y_pred = cls_pred_ids.reshape(-1)

        total = y_true.shape[0]
        correct = int((y_pred == y_true).sum())
        acc = correct / total if total > 0 else float("nan")  

        num_classes = int(cls_pred_logits.shape[-1])
        cm = np.zeros((num_classes, num_classes), dtype=int)  
        for t, p in zip(y_true, y_pred):
            if 0 <= t < num_classes and 0 <= p < num_classes:
                cm[t, p] += 1

        per_class_acc = []
        for c in range(num_classes):
            denom = cm[c].sum()
            per_class_acc.append((cm[c, c] / denom) if denom > 0 else np.nan)

        valid = [a for a in per_class_acc if not np.isnan(a)]
        ua = float(np.mean(valid)) if len(valid) > 0 else float("nan")

        # ctc
        ctc_pred_logits = pred.predictions[0]
        ctc_pred_ids = np.argmax(ctc_pred_logits, axis=-1)
        pred.label_ids[0][pred.label_ids[0] == -100] = processor.tokenizer.pad_token_id
        ctc_pred_str = processor.batch_decode(ctc_pred_ids)
        # we do not want to group tokens when computing the metrics
        ctc_label_str = processor.batch_decode(pred.label_ids[0], group_tokens=False)

        if logger.isEnabledFor(logging.DEBUG):
            for reference, predicted in zip(ctc_label_str, ctc_pred_str):
                logger.debug(f'reference: "{reference}"')
                logger.debug(f'predicted: "{predicted}"')
                if orthography.untransliterator is not None:
                    logger.debug(f'reference (untransliterated): "{orthography.untransliterator(reference)}"')
                    logger.debug(f'predicted (untransliterated): "{orthography.untransliterator(predicted)}"')

        wer = wer_metric.compute(predictions=ctc_pred_str, references=ctc_label_str)
        return {"acc": acc, "ua": ua, "wer": wer, "correct": correct, "total": total, "strlen": len(ctc_label_str)}

    if model_args.freeze_feature_extractor:
        model.freeze_feature_extractor()

    trainer = CTCTrainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=processor.feature_extractor,
    )

    trainable_params = []
    frozen_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params.append(name)
        else:
            frozen_params.append(name)

    logger.info("====== Trainable parameters ======")
    logger.info(f"Count: {len(trainable_params)}")
    for n in trainable_params:
        logger.info(f"[TRAIN] {n}")

    logger.info("====== Frozen parameters ======")
    logger.info(f"Count: {len(frozen_params)}")
    for n in frozen_params:
        logger.info(f"[FROZEN] {n}")

    if last_checkpoint is not None:
        checkpoint = last_checkpoint
    elif model_args.model_name_or_path is not None and os.path.isdir(model_args.model_name_or_path):
        checkpoint = model_args.model_name_or_path
    else:
        checkpoint = None

    if training_args.do_train:
        trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()

    if training_args.do_predict:
        logger.info('******* Predict ********')
        data_collator.audio_only = False
        predictions, labels, metrics = trainer.predict(val_dataset, metric_key_prefix="predict")
        logits_ctc, logits_cls = predictions  

        pred_ids = np.argmax(logits_cls, axis=-1).astype(int)
        true_labels = np.asarray(labels[1]).astype(int).reshape(-1)

        num_classes = int(logits_cls.shape[-1])

        cm = np.zeros((num_classes, num_classes), dtype=int)
        for t, p in zip(true_labels, pred_ids):
            if 0 <= t < num_classes and 0 <= p < num_classes:
                cm[t, p] += 1

        per_class_acc = []
        for c in range(num_classes):
            total_c = cm[c].sum()
            acc_c = (cm[c, c] / total_c) if total_c > 0 else np.nan
            per_class_acc.append(acc_c)

        valid_accs = [a for a in per_class_acc if not np.isnan(a)]
        ua = float(np.mean(valid_accs)) if len(valid_accs) > 0 else float("nan")

        total = int(np.sum(cm))
        wa = float(np.trace(cm) / total) if total > 0 else float("nan")

        # ========= 这里新增：确保输出目录存在 =========
        output_path = data_args.output_file              # 比如 "predictions/session1/predictions.txt"
        output_dir = os.path.dirname(output_path)
        if output_dir != "":
            os.makedirs(output_dir, exist_ok=True)
        # =========================================

        # 写文件
        with open(output_path, 'w', encoding='utf-8') as f:
            # 先写指标
            f.write(f"UA: {ua:.6f}\n")
            f.write(f"WA: {wa:.6f}\n")

            # 写各类别准确率，方便排查（可选）
            f.write("Per-class accuracy (class_id: accuracy):\n")
            for c, a in enumerate(per_class_acc):
                if np.isnan(a):
                    f.write(f"  {c}: NaN (no samples)\n")
                else:
                    f.write(f"  {c}: {a:.6f}\n")

            # 写混淆矩阵（行=真实，列=预测）
            f.write("Confusion Matrix (rows=true, cols=pred):\n")
            for r in range(num_classes):
                row_str = " ".join(str(x) for x in cm[r])
                f.write(row_str + "\n")

            # 空行分隔
            f.write("\n# filename true_label pred_label\n")

            # 逐样本写：文件名 真实情感标签 预测情感标签
            for i in range(len(pred_ids)):
                filename = val_dataset[i]['file'].split("/")[-1]
                f.write(f"{filename} {int(true_labels[i])} {int(pred_ids[i])}\n")


    elif training_args.do_eval:
        predictions, labels, metrics = trainer.predict(val_dataset, metric_key_prefix="eval")
        logits_ctc, logits_cls = predictions
        pred_ids = np.argmax(logits_cls, axis=-1)
        correct = np.sum(pred_ids == labels[1])
        acc = correct / len(pred_ids)
        print('correct:', correct, ', acc:', acc)


if __name__ == "__main__":
    try:
        main()
    finally:
        # 在多卡/多机时，确保所有通信完成后再销毁进程组
        if dist.is_available() and dist.is_initialized():
            try:
                dist.barrier()
            finally:
                dist.destroy_process_group()