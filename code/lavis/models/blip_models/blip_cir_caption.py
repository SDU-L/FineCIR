"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import torch
import torch.nn.functional as F
from lavis.common.registry import registry
from torch import nn
from lavis.models.blip_models.blip import BlipBase
from lavis.models.blip_models.blip_outputs import (
    BlipOutput,
    BlipIntermediateOutput,
)

from lavis.models.blip_models.blip_outputs import BlipOutputFeatures
from lavis.models.med import XBertEncoder
from lavis.models.med import XBertLMHeadDecoder
from lavis.models.vit import VisionTransformerEncoder

@registry.register_model("BlipCirCaption")
class BlipCirCaption(BlipBase):
    """
    BLIP captioning model.

    Supported model types:
        - base_coco: fine-tuned BLIP base model on COCO caption dataset (Karparthy split).
        - large_coco: fine-tuned BLIP large model on COCO caption dataset (Karparthy split).

    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("blip_caption", "base_coco")
        >>> model = load_model("blip_caption", "large_coco")
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "base": "configs/models/blip_cir_caption.yaml",
        "large": "configs/models/blip_cir_large.yaml",
        "base_coco": "configs/models/blip_cir_caption_base_coco.yaml",
        "large_coco": "configs/models/blip_caption_large_coco.yaml",
    }

    def __init__(self, image_encoder, text_decoder, text_encoder, prompt=None, max_txt_len=40):
        super().__init__()

        self.tokenizer = self.init_tokenizer()

        self.visual_encoder = image_encoder
        self.text_decoder = text_decoder
        self.text_encoder = text_encoder
        text_width = self.text_encoder.config.hidden_size
        vision_width = image_encoder.vision_width
        embed_dim=256
        self.vision_proj = nn.Linear(vision_width, embed_dim)
        self.text_proj = nn.Linear(text_width, embed_dim)

        self.max_txt_len = max_txt_len

        self.temp = nn.Parameter(0.07 * torch.ones([]))
        self.prompt = prompt
        self.prompt_length = len(self.tokenizer(self.prompt).input_ids) - 1

        self.max_txt_len = max_txt_len

    def forward_encoder(self, samples):
        image_embeds = self.visual_encoder.forward_features(samples["image"])
        return image_embeds

    def forward_decoder(self, samples, image_embeds):
        # prepare inputs for forwarding decoder
        raw_text = samples["text_input"]
        text = self.tokenizer(
            raw_text,
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(self.device)
        text.input_ids[:, 0] = self.tokenizer.bos_token_id

        # prepare targets for forwarding decoder
        decoder_targets = text.input_ids.masked_fill(
            text.input_ids == self.tokenizer.pad_token_id, -100
        )
        decoder_targets[:, : self.prompt_length] = -100

        # forward decoder
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            self.device
        )
        decoder_output = self.text_decoder(
            input_ids=text.input_ids,
            attention_mask=text.attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            labels=decoder_targets,
            return_dict=True,
        )

        return decoder_output, decoder_targets
    

    def forward(self, samples):
        image = samples["image"]
        target = samples["target"]
        caption = samples["text_input"]
        fusion_feats = self.extract_retrieval_compose(image, caption)
        target_feats = self.extract_retrieval_target(target)

        # print(caption)
        # print(self.generate_(image_embeds))
        # print(self.generate_(text_embeds))
        # print(self.generate_(fusion_output.last_hidden_state))
        # print(self.generate_(target_embeds))

        #print('target feats:', target_feats.size())
        sim_i2t = fusion_feats @ target_feats.t()
        # sim_t2q = torch.matmul(
        #     fusion_feats.unsqueeze(1).unsqueeze(1), target_feats.permute(0, 2, 1)
        # ).squeeze()

        # text-image similarity: aggregate across all query tokens
        bs = image.size(0)
        targets = torch.linspace(0,  bs - 1, bs, dtype=int).to(
            image.device
        )
        # sim_i2t, _ = sim_t2q.max(-1)
        # sim_i2t, _ = torch.topk(sim_t2q, k=5, dim=-1)
        # sim_i2t = sim_i2t.mean(-1)
        sim_i2t = sim_i2t / self.temp
        loss_itc = F.cross_entropy(sim_i2t, targets)
        return {'loss_itc': loss_itc}

        # return {'loss_itc': 0}


    #@torch.no_grad()
    def extract_retrieval_compose(self, img, mod):
        image_embeds = self.visual_encoder.forward_features(img)
        # ref_caption = self.generate_(image_embeds)
        # fusion_caption = []
        # for i in range(len(ref_caption)):
        #     f_caption = "Change '{0}' by the text '{1}' .".format(ref_caption[i], mod[i])
        #     fusion_caption.append(f_caption)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            self.device
        )

        text = self.tokenizer(mod, return_tensors="pt", padding=True).to(
            self.device
        )
        text.input_ids[:, 0] = self.tokenizer.enc_token_id

        fusion_output = self.text_encoder(
            input_ids=text.input_ids,
            attention_mask=text.attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )


        fusion_feats = F.normalize(
            self.text_proj(fusion_output.last_hidden_state[:, 0, :]), dim=-1
        )

        return fusion_feats#.unsqueeze(1).unsqueeze(1)

    #@torch.no_grad()
    def extract_retrieval_target(self, img):
        target_embeds = self.visual_encoder.forward_features(img)[:, 0, :]

        target_features = self.vision_proj(target_embeds)
        target_feats = F.normalize(target_features, dim=-1)
        return target_feats#.permute(0, 2, 1)




    def forward_(self, samples):
        r"""
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
                - text_input (list): A list of strings of length batch_size.
        Returns:
            output (BlipOutput): A BlipOutput object containing the following
                attributes:
                - loss (torch.Tensor): A scalar tensor containing the total loss. For BlipCaption, this is the same as the LM loss.
                - loss_lm (torch.Tensor): A scalar tensor containing the LM loss.
                - intermediate_outputs (BlipIntermediateOutput): A BlipIntermediateOutput object containing intermediate outputs.
                  see :class:`lavis.models.blip_models.blip_outputs.BlipOutput` for more details.

        Example:
        ```python
        >>> from PIL import Image
        >>> from lavis.models import load_model_and_preprocess
        >>> model, vis_processors, txt_processors = load_model_and_preprocess("blip_caption")
        >>> raw_image = Image.open("docs/data/merlion.png").convert("RGB")
        >>> image = vis_processors["eval"](raw_image).unsqueeze(0)
        >>> text_input = ["a large statue of a person spraying water from a fountain"]
        >>> samples = {"image": image, "text_input": text_input}
        >>> output = model(samples)
        >>> output.keys()
        odict_keys(['intermediate_output', 'loss', 'loss_lm'])
        >>> output.intermediate_output.image_embeds.shape
        torch.Size([1, 577, 768])
        >>> output.intermediate_output.decoder_labels.shape
        torch.Size([1, 13])
        ```"""

        image_embeds = self.forward_encoder(samples)
        decoder_output, decoder_targets = self.forward_decoder(samples, image_embeds)

        # return decoder_out
        return BlipOutput(
            loss=decoder_output.loss,
            loss_lm=decoder_output.loss,
            intermediate_output=BlipIntermediateOutput(
                image_embeds=image_embeds,
                decoder_output=decoder_output,
                decoder_labels=decoder_targets,
            ),
        )
    
    @torch.no_grad()
    def generate_(
        self,
        encoder_out,
        use_nucleus_sampling=False,
        num_beams=3,
        max_length=30,
        min_length=10,
        top_p=0.9,
        repetition_penalty=1.0,
        num_captions=1,
    ):
        # prepare inputs for decoder generation.
        #encoder_out = self.forward_encoder(samples)
        image_embeds = torch.repeat_interleave(encoder_out, num_captions, 0)

        prompt = [self.prompt] * image_embeds.size(0)
        prompt = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt.input_ids[:, 0] = self.tokenizer.bos_token_id
        prompt.input_ids = prompt.input_ids[:, :-1]

        # get decoded text
        decoder_out = self.text_decoder.generate_from_encoder(
            tokenized_prompt=prompt,
            visual_embeds=image_embeds,
            sep_token_id=self.tokenizer.sep_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            use_nucleus_sampling=use_nucleus_sampling,
            num_beams=num_beams,
            max_length=max_length,
            min_length=min_length,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )

        outputs = self.tokenizer.batch_decode(decoder_out, skip_special_tokens=True)
        captions = [output[len(self.prompt) :] for output in outputs]

        return captions

    def generate(
        self,
        samples,
        use_nucleus_sampling=False,
        num_beams=3,
        max_length=30,
        min_length=10,
        top_p=0.9,
        repetition_penalty=1.0,
        num_captions=1,
    ):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
            use_nucleus_sampling (bool): Whether to use nucleus sampling. If False, use top-k sampling.
            num_beams (int): Number of beams for beam search. 1 means no beam search.
            max_length (int): The maximum length of the sequence to be generated.
            min_length (int): The minimum length of the sequence to be generated.
            top_p (float): The cumulative probability for nucleus sampling.
            repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty.
            num_captions (int): Number of captions to be generated for each image.
        Returns:
            captions (list): A list of strings of length batch_size * num_captions.

        Example:
        ```python
        >>> from PIL import Image
        >>> from lavis.models import load_model_and_preprocess
        >>> model, vis_processors, txt_processors = load_model_and_preprocess("blip_caption")
        >>> raw_image = Image.open("docs/data/merlion.png").convert("RGB")
        >>> image = vis_processors["eval"](raw_image).unsqueeze(0)
        >>> samples = {"image": image}
        >>> captions = model.generate(samples)
        >>> captions
        ['a large statue of a person spraying water from a fountain']
        >>> captions = model.generate(samples, use_nucleus_sampling=True, num_captions=3)
        >>> captions # example output, results may vary due to randomness
        ['singapore showing the view of some building',
        'the singapore harbor in twilight, as the weather is going down',
        'the famous singapore fountain at sunset']
        """
        # prepare inputs for decoder generation.
        encoder_out = self.forward_encoder(samples)
        image_embeds = torch.repeat_interleave(encoder_out, num_captions, 0)

        prompt = [self.prompt] * image_embeds.size(0)
        prompt = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt.input_ids[:, 0] = self.tokenizer.bos_token_id
        prompt.input_ids = prompt.input_ids[:, :-1]

        # get decoded text
        decoder_out = self.text_decoder.generate_from_encoder(
            tokenized_prompt=prompt,
            visual_embeds=image_embeds,
            sep_token_id=self.tokenizer.sep_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            use_nucleus_sampling=use_nucleus_sampling,
            num_beams=num_beams,
            max_length=max_length,
            min_length=min_length,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )

        outputs = self.tokenizer.batch_decode(decoder_out, skip_special_tokens=True)
        captions = [output[len(self.prompt) :] for output in outputs]

        return captions

    @classmethod
    def from_config(cls, cfg):
        # vision encoder
        image_encoder = VisionTransformerEncoder.from_config(cfg)
        # text encoder + multimodal decoder
        text_decoder = XBertLMHeadDecoder.from_config(cfg)
        text_encoder = XBertEncoder.from_config(cfg)

        prompt = cfg.get("prompt", None)
        max_txt_len = cfg.get("max_txt_len", 40)

        model = cls(image_encoder, text_decoder,text_encoder, prompt=prompt, max_txt_len=max_txt_len)
        # model.load_checkpoint_from_config(cfg)
        pretrain_path = cfg.get("pretrained", None)
        if pretrain_path is not None:
            msg = model.load_from_pretrained(url_or_filename=pretrain_path)
        return model
