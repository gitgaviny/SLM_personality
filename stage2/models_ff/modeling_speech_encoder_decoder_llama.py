from typing import Optional, Tuple, Union
from dataclasses import dataclass
import torch.nn.functional as F
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
import numpy as np
import os
import sys
parent = os.path.abspath(os.path.join(__file__, "..", ".."))
if parent not in sys.path:
    sys.path.insert(0, parent)
from utils.generation_utils import GenerationMixin_Instruct

from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from transformers.models.auto.configuration_auto import AutoConfig
from transformers.models.auto.modeling_auto import AutoModel, AutoModelForCausalLM
from transformers.models.speech_encoder_decoder.configuration_speech_encoder_decoder import SpeechEncoderDecoderConfig

from modeling_llama import LlamaForCausalLM

import logging
logger = logging.getLogger(__name__)

_CONFIG_FOR_DOC = "SpeechEncoderDecoderConfig"

@dataclass
class SpeechSeq2SeqLMOutput(Seq2SeqLMOutput):
    emotion_logits: Optional[torch.FloatTensor] = None  
    
# Copied from transformers.models.encoder_decoder.modeling_encoder_decoder.shift_tokens_right
def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    if decoder_start_token_id is None:
        raise ValueError("Make sure to set the decoder_start_token_id attribute of the model's configuration.")
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("Make sure to set the pad_token_id attribute of the model's configuration.")
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids


class WavLMAdapterLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.conv = nn.Conv1d(
            config.hidden_size,
            2 * config.hidden_size,
            kernel_size=3,
            stride=2,
            padding=1,
        )

    def forward(self, hidden_states):
        hidden_states = self.conv(hidden_states)
        hidden_states = nn.functional.glu(hidden_states, dim=1)

        return hidden_states


class WavLMAdapter(nn.Module):
    def __init__(self, config):
        super().__init__()

        # feature dim might need to be down-projected
        # if config.output_hidden_size != config.hidden_size:
        #     self.proj = nn.Linear(config.hidden_size, config.output_hidden_size)
        #     self.proj_layer_norm = nn.LayerNorm(config.output_hidden_size)
        # else:
        self.proj = self.proj_layer_norm = None

        self.layers = nn.ModuleList(WavLMAdapterLayer(config) for _ in range(3))
        self.layerdrop = config.layerdrop


    def forward(self, hidden_states):
        # down project hidden_states if necessary
        if self.proj is not None and self.proj_layer_norm is not None:
            hidden_states = self.proj(hidden_states)
            hidden_states = self.proj_layer_norm(hidden_states)

        hidden_states = hidden_states.transpose(1, 2)

        for layer in self.layers:
            layerdrop_prob = np.random.random()
            if not self.training or (layerdrop_prob > self.layerdrop):
                hidden_states = layer(hidden_states)

        hidden_states = hidden_states.transpose(1, 2)
        return hidden_states


class SpeechEncoderDecoderModelLlama(PreTrainedModel, GenerationMixin_Instruct):
    config_class = SpeechEncoderDecoderConfig
    base_model_prefix = "speech_encoder_decoder"
    main_input_name = "inputs"
    supports_gradient_checkpointing = True
    _supports_param_buffer_assignment = False
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    def __init__(
        self,
        config: Optional[PretrainedConfig] = None,
        encoder: Optional[PreTrainedModel] = None,
        decoder: Optional[PreTrainedModel] = None,
    ):
        if config is None and (encoder is None or decoder is None):
            raise ValueError("Either a configuration or an encoder and a decoder has to be provided.")
        if config is None:
            config = SpeechEncoderDecoderConfig.from_encoder_decoder_configs(encoder.config, decoder.config)
        else:
            if not isinstance(config, self.config_class):
                raise ValueError(f"Config: {config} has to be of type {self.config_class}")

        if config.decoder.cross_attention_hidden_size is not None:
            if config.decoder.cross_attention_hidden_size != config.encoder.hidden_size:
                raise ValueError(
                    "If `cross_attention_hidden_size` is specified in the decoder's configuration, it has to be equal"
                    f" to the encoder's `hidden_size`. Got {config.decoder.cross_attention_hidden_size} for"
                    f" `config.decoder.cross_attention_hidden_size` and {config.encoder.hidden_size} for"
                    " `config.encoder.hidden_size`."
                )

        # initialize with config
        # make sure input & output embeddings is not tied
        config.tie_word_embeddings = False
        super().__init__(config)

        if encoder is None:
            encoder = AutoModel.from_config(config.encoder)

        if decoder is None:
            decoder = LlamaForCausalLM._from_config(config.decoder)

        self.encoder = encoder
        self.decoder = decoder

        self.cls_head = nn.Linear(self.decoder.config.hidden_size *2 , 3)
        self.wavlm_adapter = WavLMAdapter(self.encoder.config)

        if self.encoder.config.to_dict() != self.config.encoder.to_dict():
            logger.warning(
                f"Config of the encoder: {self.encoder.__class__} is overwritten by shared encoder config:"
                f" {self.config.encoder}"
            )
        if self.decoder.config.to_dict() != self.config.decoder.to_dict():
            logger.warning(
                f"Config of the decoder: {self.decoder.__class__} is overwritten by shared decoder config:"
                f" {self.config.decoder}"
            )

        # make sure that the individual model's config refers to the shared config
        # so that the updates to the config will be synced
        self.config.encoder._attn_implementation = self.encoder.config._attn_implementation
        self.config.decoder._attn_implementation = self.decoder.config._attn_implementation
        self.encoder.config = self.config.encoder
        self.decoder.config = self.config.decoder

        # For special tokens
        self.ignore_token_id = getattr(self.config, "ignore_token_id", None)
        self.pad_token_id    = getattr(self.config, "pad_token_id",   None)
        self.sc_token_id     = getattr(self.config, "sc_token_id",    None)
        self.talker_ctc      = getattr(self.config, "talker_ctc",     False)
        self.talker_numbers  = getattr(self.config, "talker_numbers", 2)
        self.instruct        = getattr(self.config, "instruct",       False)
        self.eos_token_id    = getattr(self.config, "eos_token_id", None)

        if self.instruct:
            self.bosp_token_id = self.config.bosp_token_id
            self.bosr_token_id = self.config.bosr_token_id
            self.boss_token_id = self.config.boss_token_id
            self.eosp_token_id = self.config.eosp_token_id
            self.eosr_token_id = self.config.eosr_token_id
            self.eoss_token_id = self.config.eoss_token_id

        # get encoder output hidden size
        self.encoder_output_dim = getattr(config.encoder, "output_hidden_size", config.encoder.hidden_size)
        if (
            self.encoder_output_dim != self.decoder.config.hidden_size
            and self.decoder.config.cross_attention_hidden_size is None
        ):
            # encoder outputs might need to be projected to different dimension for decoder
            self.enc_to_dec_proj = nn.Linear(self.encoder.config.hidden_size, self.decoder.config.hidden_size)

        if self.encoder.get_output_embeddings() is not None:
            raise ValueError(
                f"The encoder {self.encoder} should not have a LM Head. Please use a model without LM Head"
            )

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def get_input_embeddings(self):
        return self.decoder.get_input_embeddings()

    def get_output_embeddings(self):
        return self.decoder.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        return self.decoder.set_output_embeddings(new_embeddings)

    def freeze_feature_encoder(self):
        self.encoder.feature_extractor._freeze_parameters()

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)

    def resize_token_embeddings(self, *args, **kwargs):
        raise NotImplementedError(
            "Resizing the embedding layers via the SpeechEncoderDecoderModel directly is not supported. Please use the"
            " respective methods of the wrapped decoder object (model.decoder.resize_token_embeddings(...))"
        )

    def _reorder_cache(self, past_key_values, beam_idx):
        # apply decoder cache reordering here
        return self.decoder._reorder_cache(past_key_values, beam_idx)

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        # At the moment fast initialization is not supported for composite models
        if kwargs.get("_fast_init", False):
            logger.warning(
                "Fast initialization is currently not supported for SpeechEncoderDecoderModel. "
                "Falling back to slow initialization..."
            )
        kwargs["_fast_init"] = False
        return super().from_pretrained(*args, **kwargs)

    @classmethod
    def from_encoder_decoder_pretrained(
        cls,
        encoder_pretrained_model_name_or_path: str = None,
        decoder_pretrained_model_name_or_path: str = None,
        *model_args,
        **kwargs,
    ) -> PreTrainedModel:
        r"""
        Instantiate an encoder and a decoder from one or two base classes of the library from pretrained model
        checkpoints.
        Example:

        ```python
        >>> from transformers import SpeechEncoderDecoderModel

        >>> # initialize a wav2vec2bert from a pretrained Wav2Vec2 and a pretrained BERT model. Note that the cross-attention layers will be randomly initialized
        >>> model = SpeechEncoderDecoderModel.from_encoder_decoder_pretrained(
        ...     "facebook/wav2vec2-base-960h", "google-bert/bert-base-uncased"
        ... )
        >>> # saving model after fine-tuning
        >>> model.save_pretrained("./wav2vec2bert")
        >>> # load fine-tuned model
        >>> model = SpeechEncoderDecoderModel.from_pretrained("./wav2vec2bert")
        ```"""
        kwargs_encoder = {
            argument[len("encoder_") :]: value for argument, value in kwargs.items() if argument.startswith("encoder_")
        }

        kwargs_decoder = {
            argument[len("decoder_") :]: value for argument, value in kwargs.items() if argument.startswith("decoder_")
        }

        # remove encoder, decoder kwargs from kwargs
        for key in kwargs_encoder.keys():
            del kwargs["encoder_" + key]
        for key in kwargs_decoder.keys():
            del kwargs["decoder_" + key]

        # Load and initialize the encoder and decoder
        # The distinction between encoder and decoder at the model level is made
        # by the value of the flag `is_decoder` that we need to set correctly.
        encoder = kwargs_encoder.pop("model", None)
        if encoder is None:
            if encoder_pretrained_model_name_or_path is None:
                raise ValueError(
                    "If `encoder_model` is not defined as an argument, a `encoder_pretrained_model_name_or_path` has "
                    "to be defined."
                )

            if "config" not in kwargs_encoder:
                encoder_config, kwargs_encoder = AutoConfig.from_pretrained(
                    encoder_pretrained_model_name_or_path, **kwargs_encoder, return_unused_kwargs=True
                )

                if encoder_config.is_decoder is True or encoder_config.add_cross_attention is True:
                    logger.info(
                        f"Initializing {encoder_pretrained_model_name_or_path} as a encoder model "
                        "from a decoder model. Cross-attention and casual mask are disabled."
                    )
                    encoder_config.is_decoder = False
                    encoder_config.add_cross_attention = False

                kwargs_encoder["config"] = encoder_config

            encoder = AutoModel.from_pretrained(encoder_pretrained_model_name_or_path, *model_args, **kwargs_encoder)

        decoder = kwargs_decoder.pop("model", None)
        if decoder is None:
            if decoder_pretrained_model_name_or_path is None:
                raise ValueError(
                    "If `decoder_model` is not defined as an argument, a `decoder_pretrained_model_name_or_path` has "
                    "to be defined."
                )

            if "config" not in kwargs_decoder:
                decoder_config, kwargs_decoder = AutoConfig.from_pretrained(
                    decoder_pretrained_model_name_or_path, **kwargs_decoder, return_unused_kwargs=True
                )

                if decoder_config.is_decoder is False or decoder_config.add_cross_attention is False:
                    logger.info(
                        f"Initializing {decoder_pretrained_model_name_or_path} as a decoder model. Cross attention"
                        f" layers are added to {decoder_pretrained_model_name_or_path} and randomly initialized if"
                        f" {decoder_pretrained_model_name_or_path}'s architecture allows for cross attention layers."
                    )
                    decoder_config.is_decoder = True
                    decoder_config.add_cross_attention = True

                kwargs_decoder["config"] = decoder_config

            if kwargs_decoder["config"].is_decoder is False or kwargs_decoder["config"].add_cross_attention is False:
                logger.warning(
                    f"Decoder model {decoder_pretrained_model_name_or_path} is not initialized as a decoder. "
                    f"In order to initialize {decoder_pretrained_model_name_or_path} as a decoder, "
                    "make sure that the attributes `is_decoder` and `add_cross_attention` of `decoder_config` "
                    "passed to `.from_encoder_decoder_pretrained(...)` are set to `True` or do not pass a "
                    "`decoder_config` to `.from_encoder_decoder_pretrained(...)`"
                )

            decoder = AutoModelForCausalLM.from_pretrained(decoder_pretrained_model_name_or_path, **kwargs_decoder)

        # instantiate config with corresponding kwargs
        config = SpeechEncoderDecoderConfig.from_encoder_decoder_configs(encoder.config, decoder.config, **kwargs)

        # make sure input & output embeddings is not tied
        config.tie_word_embeddings = False
        return cls(encoder=encoder, decoder=decoder, config=config)

    def generate(self, *args, **kwargs):
        """
        Forward every call to the mix-in’s version explicitly.
        """
        return GenerationMixin_Instruct.generate(self, *args, **kwargs)

    @staticmethod
    def _expand_inputs_for_generation(*args, **kwargs):
        # delegate to the mix-in’s staticmethod
        return GenerationMixin_Instruct._expand_inputs_for_generation(*args, **kwargs)

    def _sample(
        self,
        input_ids,
        logits_processor,
        stopping_criteria,
        generation_config,
        synced_gpus,
        streamer=None,
        **model_kwargs,
    ):
        """
        Forward every call to the mix-in’s custom `_sample`.
        """
        return GenerationMixin_Instruct._sample(
            self,
            input_ids=input_ids,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            generation_config=generation_config,
            synced_gpus=synced_gpus,
            streamer=streamer,
            **model_kwargs,
        )

    def forward(
        self,
        inputs: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        prompt_ids: Optional[torch.LongTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        encoder_outputs: Optional[Tuple[torch.FloatTensor]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        input_values: Optional[torch.FloatTensor] = None,
        input_features: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        return_emotion_logits: Optional[bool] = False,
        **kwargs,
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            MARKER_ID = 128264
            EMOTION_BASE = 128257  

            # labels: [B, L]
            B, L = labels.shape
            marker_mask = (labels == MARKER_ID)
            marker_indices = marker_mask.nonzero(as_tuple=False) 
            marker_indices = marker_indices[marker_indices[:, 0].argsort()]

            idx_marker = marker_indices[:, 1]                 # [B]
            idx_emotion = idx_marker + 1 
            batch_indices = torch.arange(B, device=labels.device)
            emotion_labels = labels[batch_indices, idx_emotion] - EMOTION_BASE  # [B]
            keep_mask = torch.ones_like(labels, dtype=torch.bool) 
            keep_mask[batch_indices, idx_emotion] = False       
            labels = labels[keep_mask].view(B, L - 1)

        kwargs_encoder = {argument: value for argument, value in kwargs.items() if not argument.startswith("decoder_")}

        kwargs_decoder = {
            argument[len("decoder_") :]: value for argument, value in kwargs.items() if argument.startswith("decoder_")
        }
        if "num_items_in_batch" in kwargs_encoder:
            kwargs_decoder["num_items_in_batch"] = kwargs_encoder.pop("num_items_in_batch", None)

        if encoder_outputs is None:
            if inputs is None:
                if input_values is not None and input_features is not None:
                    raise ValueError("You cannot specify both input_values and input_features at the same time")
                elif input_values is not None:
                    inputs = input_values
                elif input_features is not None:
                    inputs = input_features
                else:
                    raise ValueError("You have to specify either input_values or input_features")

            encoder_outputs = self.encoder(
                inputs,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs_encoder,
            )
        elif isinstance(encoder_outputs, tuple):
            encoder_outputs = BaseModelOutput(*encoder_outputs)

        encoder_hidden_states = encoder_outputs[0]          # (B, T_feat, H)
        orig_seq_len = encoder_hidden_states.shape[1]

        if attention_mask is not None:
            orig_encoder_attention_mask = self.encoder._get_feature_vector_attention_mask(
                orig_seq_len, attention_mask
            )                                             # (B, T_feat)
        else:
            orig_encoder_attention_mask = None

        encoder_hidden_states = self.wavlm_adapter(encoder_hidden_states)  # (B, T_feat', H)
        new_seq_len = encoder_hidden_states.shape[1]

        if (
            self.encoder_output_dim != self.decoder.config.hidden_size
            and self.decoder.config.cross_attention_hidden_size is None
        ):
            encoder_hidden_states = self.enc_to_dec_proj(encoder_hidden_states)

        if orig_encoder_attention_mask is not None:
            downsample_factor = orig_seq_len // new_seq_len
            if downsample_factor * new_seq_len != orig_seq_len:
                encoder_attention_mask = F.interpolate(
                    orig_encoder_attention_mask.unsqueeze(1).float(),  # (B, 1, T_feat)
                    size=new_seq_len,
                    mode="nearest",
                ).squeeze(1).to(dtype=orig_encoder_attention_mask.dtype)  # (B, T_feat')
            else:
                encoder_attention_mask = orig_encoder_attention_mask[:, ::downsample_factor]
        else:
            encoder_attention_mask = None


        if (labels is not None) and (decoder_input_ids is None and decoder_inputs_embeds is None):
            # labels-> decoder_input_ids : add  
            decoder_input_ids = shift_tokens_right(
                labels, self.config.pad_token_id, self.config.decoder_start_token_id
            )

            # Here we make the speech-padded labels: with self.ignore_token_id (-100)
            batch, speech_len, _ = encoder_hidden_states.shape

            # Insert <eos>
            # For input_ids, the <eos> should not be inserted, the <pad> is inserted
            # <eos> is only inserted into labels
            decoder_ids_pad = torch.full((batch, 1), self.pad_token_id, device=labels.device)
            decoder_input_ids = torch.cat((decoder_input_ids, decoder_ids_pad), dim=1)
            pad_eos = torch.full((batch, 1), self.ignore_token_id, device=labels.device)
            labels = torch.cat((labels, pad_eos), dim=1)

            mask = (labels == self.ignore_token_id) # self.ignore_token_id = -100
            first_pad_id = mask.float().argmax(dim=1)
            # Here we add the <eos> in labels
            labels[torch.arange(batch), first_pad_id] = eos_token_id = self.config.eos_token_id[0] if isinstance(self.config.eos_token_id, (list, tuple)) else self.config.eos_token_id

            # Compute the length of prompt
            # !!!! Should be fixed a little bit later
            # Currently, since all the prompts are same, we directly use one sample to compute the length
            # TODO: But should be modifed for variable-prompt condition
            if self.instruct:
                seq = labels[0]
                len_prompt = (seq.eq(self.eosp_token_id).nonzero()[0] - seq.eq(self.bosp_token_id).nonzero()[0] - 1).item()

                # The ignore part during computing loss:
                # (<bos_prompt>, prompt, <eos_prompt>, <bos_speech>, speech_emb, <eos_speech>, <bos_response>)
                # The corresponding length is:
                # (1, prompt_length, 1, 1, speech_length, 1, 1) => prompt_length + speech_length + 5
                # Currently, the labels is:
                # (<bos_prompt>, prompt, <eos_prompt>, <bos_speech>, speech_emb, <eos_speech>, <bos_response>, transcription)
                # For simple processing, we directly generate a mask for the above part (without transcription)
                # and use the text contents (after <bos_response>)
                ignore_contents_mask = torch.full((batch, speech_len + len_prompt + 5), self.config.ignore_token_id, device=labels.device)
                bos_response_idx = labels[0].eq(self.bosr_token_id).nonzero()[0]
                contents = labels[:, bos_response_idx+1:]
                labels = torch.cat((ignore_contents_mask, contents), dim=1)

            else:
                # The ignore part during computing loss: 
                # (speech_emb)
                # The corresponding length is:
                # (speech_length) => speech_length
                # Currently, the labels is:
                # (prompt, text)
                # For simple processing, we directly generate a mask for the above part
                ignore_contents_mask = torch.full((batch, speech_len), self.config.ignore_token_id, device=labels.device)
                labels = torch.cat((ignore_contents_mask, labels), dim=1)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            use_cache=use_cache,
            past_key_values=past_key_values,
            return_dict=return_dict,
            **kwargs_decoder,
        )

        decoder_utt = torch.mean(decoder_outputs.hidden_states, dim=1)
        encoder_utt = torch.mean(encoder_hidden_states, dim=1)
        emotion = torch.cat([decoder_utt, encoder_utt], dim=1)
        emotion = self.cls_head(emotion)

        # Compute loss independent from decoder (as some shift the logits inside them)
        loss = None

        if labels is not None:
            logits = decoder_outputs.logits if return_dict else decoder_outputs[0]
            loss_fct = CrossEntropyLoss()
            loss_llama = loss_fct(logits.reshape(-1, self.decoder.config.vocab_size), labels.reshape(-1))
            loss_emotion = loss_fct(emotion, emotion_labels)
            loss = loss_llama + loss_emotion
            # loss = loss_emotion

        if not return_dict:
            if loss is not None:
                return (loss,) + decoder_outputs + encoder_outputs
            else:
                return decoder_outputs + encoder_outputs

        return SpeechSeq2SeqLMOutput(
            loss=loss,
            logits=decoder_outputs.logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            encoder_last_hidden_state=encoder_hidden_states,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
            emotion_logits = emotion,
        )

__all__ = ["SpeechEncoderDecoderModelLlama"]
