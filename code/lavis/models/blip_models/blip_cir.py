"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import warnings

import torch
import torch.nn.functional as F
from lavis.common.registry import registry
from lavis.models.blip_models.blip import BlipBase
from lavis.models.blip_models.blip_outputs import BlipOutputFeatures
from lavis.models.med import XBertEncoder
from lavis.models.vit import VisionTransformerEncoder
from torch import nn


@registry.register_model("BlipCir")
class BlipCir(BlipBase):
    """
    Class for BLIP feature extractor.

    Supported model types:
        - base: BLIP base model with pre-trained weights from capfilt by BLIP large model.

    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("blip_feature_extractor", "base")
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "base": "configs/models/blip_feature_extractor_base.yaml",
        "large": "configs/models/blip_cir_large.yaml",
        #"base": "configs/models/blip_itm_base.yaml",
        # "large": "configs/models/blip_feature_extractor_large.yaml",
    }

    def __init__(self, image_encoder, text_encoder, embed_dim, max_txt_len=40):
        super().__init__()

        self.tokenizer = self.init_tokenizer()

        self.visual_encoder = image_encoder
        self.text_encoder = text_encoder

        # creating projection layers for ITC
        text_width = text_encoder.config.hidden_size
        vision_width = image_encoder.vision_width

        self.vision_proj = nn.Linear(vision_width, embed_dim)
        self.text_proj = nn.Linear(text_width, embed_dim)

        self.max_txt_len = max_txt_len

        self.temp = nn.Parameter(0.07 * torch.ones([]))


    def forward(self, samples):
        image = samples["image"]
        target = samples["target"]
        caption = samples["text_input"]

        ###============== reference text fusion ===================###
        image_embeds = self.visual_encoder.forward_features(image)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            self.device
        )

        text = self.tokenizer(caption, return_tensors="pt", padding=True).to(
            self.device
        )
        text.input_ids[:, 0] = self.tokenizer.enc_token_id

        fusion_output = self.text_encoder(
            text.input_ids,
            attention_mask=text.attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        #multimodal_embeds = output.last_hidden_state

        fusion_feats = F.normalize(
            self.text_proj(fusion_output.last_hidden_state[:, 0, :]), dim=-1
        )
        #print('fusion feats:', fusion_feats.size())

        ###============== Fusion-target Contrastive ===================###
        # reference image feature  
        target_embeds = self.visual_encoder.forward_features(target)[:, 0, :]

        target_features = self.vision_proj(target_embeds)
        target_feats = F.normalize(target_features, dim=-1)

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


    @torch.no_grad()
    def extract_retrieval_compose(self, img, mod):
        image_embeds = self.visual_encoder.forward_features(img)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            self.device
        )

        text = self.tokenizer(mod, return_tensors="pt", padding=True).to(
            self.device
        )
        text.input_ids[:, 0] = self.tokenizer.enc_token_id

        fusion_output = self.text_encoder(
            text.input_ids,
            attention_mask=text.attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        #multimodal_embeds = output.last_hidden_state


        fusion_feats = F.normalize(
            self.text_proj(fusion_output.last_hidden_state[:, 0, :]), dim=-1
        )

        # text_output = self.Qformer.bert(
        #     text_tokens.input_ids,
        #     query_embeds=fusion_output.last_hidden_state[:, : query_tokens.size(1), :],
        #     attention_mask=attention_mask,
        #     return_dict=True,
        # )


        return fusion_feats#.unsqueeze(1).unsqueeze(1)

    @torch.no_grad()
    def extract_retrieval_target(self, img):
        target_embeds = self.visual_encoder.forward_features(img)[:, 0, :]

        target_features = self.vision_proj(target_embeds)
        target_feats = F.normalize(target_features, dim=-1)
        return target_feats#.permute(0, 2, 1)




    @torch.no_grad()
    def extract_features(self, samples, mode="multimodal"):
        """
        Extract features for multimodal or unimodal samples.

        Args:
            samples (dict): A dictionary of samples, containing the following keys:
                - image (torch.Tensor): A tensor of shape (B, C, H, W) containing the image.
                    Raw images should be preprocessed before being passed to feature extractor.
                - text_input (list): A list of strings containing the text, length B.
            mode (str): The mode of feature extraction. Can be either "multimodal", "text" or "image".
                If "multimodal", return image features and multimodal features;
                if "text", return text features;
                if "image", return image features.
                Default: "multimodal".

        Returns:
            BlipOutputFeatures: A BlipOutputFeatures object containing the features.
                See lavis/models/blip_models/blip_outputs.py for more details.

        Examples:
        ```python
            >>> from PIL import Image
            >>> from lavis.models import load_model_and_preprocess
            >>> raw_image = Image.open("docs/data/merlion.png").convert("RGB")
            >>> caption = "a large fountain spewing water into the air"
            >>> model, vis_processors, txt_processors = load_model_and_preprocess("blip_feature_extractor", is_eval=True)
            >>> image = vis_processors["eval"](raw_image).unsqueeze(0)
            >>> text_input = txt_processors["eval"](caption)

            >>> sample = {"image": image, "text_input": [text_input]}

            >>> features_multimodal = model.extract_features(sample)
            >>> features_multimodal.keys()
            odict_keys(['image_embeds', 'multimodal_embeds'])
            >>> features_multimodal.image_embeds.shape
            torch.Size([1, 197, 768])
            >>> features_multimodal.multimodal_embeds.shape
            torch.Size([1, 12, 768])

            >>> features_text = model.extract_features(sample, mode="text")
            >>> features_text.keys()
            odict_keys(['text_embeds', 'text_features'])
            >>> features_text.text_embeds.shape
            torch.Size([1, 12, 768])
            >>> features_text.text_features.shape
            torch.Size([1, 12, 256])

            >>> features_image = model.extract_features(sample, mode="image")
            >>> features_image.keys()
            odict_keys(['image_embeds', 'image_features'])
            >>> features_image.image_embeds.shape
            torch.Size([1, 197, 768])
            >>> features_image.image_features.shape
            torch.Size([1, 197, 256])
        ```
        """
        image = samples.get("image")
        caption = samples.get("text_input")

        # assert mode is one of "image", "text", "multimodal"
        assert mode in [
            "image",
            "text",
            "multimodal",
        ], "mode must be one of 'image', 'text', 'multimodal'"

        # initalize output
        image_embeds, text_embeds, multimodal_embeds = None, None, None
        image_features, text_features = None, None

        if mode == "image":
            assert (
                image is not None
            ), "Image is not provided for mode 'image' or 'multimodal'"
            # return image features
            image_embeds = self.visual_encoder.forward_features(image)

            image_features = self.vision_proj(image_embeds)
            image_features = F.normalize(image_features, dim=-1)

        elif mode == "text":
            assert (
                caption is not None
            ), "text input is None for mode 'text' or 'multimodal'"

            text = self.tokenizer(caption, return_tensors="pt", padding=True).to(
                self.device
            )

            # return text features
            text_output = self.text_encoder(
                text.input_ids,
                attention_mask=text.attention_mask,
                return_dict=True,
                mode="text",
            )
            text_embeds = text_output.last_hidden_state

            text_features = self.text_proj(text_embeds)
            text_features = F.normalize(text_features, dim=-1)

        elif mode == "multimodal":
            # return multimodel features
            image_embeds = self.visual_encoder.forward_features(image)
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
                self.device
            )

            text = self.tokenizer(caption, return_tensors="pt", padding=True).to(
                self.device
            )
            text.input_ids[:, 0] = self.tokenizer.enc_token_id

            output = self.text_encoder(
                text.input_ids,
                attention_mask=text.attention_mask,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

            multimodal_embeds = output.last_hidden_state

        return BlipOutputFeatures(
            image_embeds=image_embeds,
            image_embeds_proj=image_features,
            text_embeds=text_embeds,
            text_embeds_proj=text_features,
            multimodal_embeds=multimodal_embeds,
        )

    @classmethod
    def from_config(cls, cfg=None):
        # set from_pretrained=True to load weights for 'bert-base-uncased'
        image_encoder = VisionTransformerEncoder.from_config(cfg)
        text_encoder = XBertEncoder.from_config(cfg)

        embed_dim = cfg.get("embed_dim", 256)
        max_txt_len = cfg.get("max_txt_len", 30)

        model = cls(
            image_encoder=image_encoder,
            text_encoder=text_encoder,
            embed_dim=embed_dim,
            max_txt_len=max_txt_len,
        )

        # load pre-trained weights
        pretrain_path = cfg.get("pretrained", None)
        if pretrain_path is not None:
            msg = model.load_from_pretrained(url_or_filename=pretrain_path)
        else:
            warnings.warn("No pretrained weights are loaded.")

        return model
