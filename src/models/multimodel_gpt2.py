import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, AutoModelForCausalLM

# Try to import GPT2LMHeadModel, fallback to AutoModelForCausalLM if not available
try:
    from transformers import GPT2LMHeadModel
except ImportError:
    GPT2LMHeadModel = AutoModelForCausalLM


class MultimodelGPT2(nn.Module):
    def __init__(
        self,
        gpt2_model_name="gpt2",
        num_img_tokens=4,
        num_unfreeze_layers=4,
        dropout_rate=0.1,
    ):
        super().__init__()

        self.gpt2 = GPT2LMHeadModel.from_pretrained(gpt2_model_name)
        self.config = self.gpt2.config
        self.hidden_size = self.config.hidden_size
        self.num_img_tokens = num_img_tokens
        self.prefix_len = 1 + self.num_img_tokens

        self.image_projection = nn.Sequential(
            nn.Linear(2048, self.hidden_size * self.num_img_tokens),
            nn.LayerNorm(self.hidden_size * self.num_img_tokens),
        )

        self.image_dropout = nn.Dropout(dropout_rate)
        self.learned_bos_embedding = nn.Parameter(torch.randn(1, 1, self.hidden_size))
        self._freeze_layers(num_unfreeze_layers)

    def _freeze_layers(self, num_unfreeze_layers):
        for param in self.gpt2.parameters():
            param.requires_grad = False
        # Unfreeze Bridge, BOS, Head, and Top Layers
        self.learned_bos_embedding.requires_grad = True

        for param in self.image_projection.parameters():
            param.requires_grad = True
        for i in range(self.config.n_layer - num_unfreeze_layers, self.config.n_layer):
            for param in self.gpt2.transformer.h[i].parameters():
                param.requires_grad = True
        for param in self.gpt2.lm_head.parameters():
            param.requires_grad = True

    def forward(self, img_features, input_ids, attention_mask=None, labels=None):

        batch_size = img_features.shape[0]

        text_embeds = self.gpt2.transformer.wte(input_ids) 
    
        img_embeds = self.image_projection(img_features).view(batch_size, self.num_img_tokens, self.hidden_size) # Expected: [B, 4, 768]
        img_embeds = self.image_dropout(img_embeds)

        bos_embeds = self.learned_bos_embedding.repeat(batch_size, 1, 1)

        full_embeds = torch.cat([bos_embeds, img_embeds, text_embeds], dim=1)
        
        if attention_mask is not None:
            prefix_mask = torch.ones((batch_size, 5), device=img_features.device)
            full_mask = torch.cat([prefix_mask, attention_mask], dim=1)
        else:
            full_mask = None

        full_labels = None
        if labels is not None:
            prefix_labels = torch.full((batch_size, self.prefix_len), -100, device=labels.device)
            full_labels = torch.cat([prefix_labels, labels], dim=1)
            
        outputs = self.gpt2(
            inputs_embeds=full_embeds,
            attention_mask=full_mask,
            labels=full_labels,
            return_dict=True,
        )

        return outputs.loss, outputs.logits[:, self.prefix_len:, :]

    def generate(
        self,
        img_features,
        input_ids=None,
        attention_mask=None,
        **gen_kwargs,
    ):

        B = img_features.shape[0]

        bos_embeds = self.learned_bos_embedding.repeat(B, 1, 1)
        img_embeds = self.image_projection(img_features).view(
            B, self.num_img_tokens, self.hidden_size
        )
        prefix_embeds = torch.cat([bos_embeds, img_embeds], dim=1)
        prompt_len = input_ids.shape[1] if input_ids is not None else 0

        if input_ids is not None:
            promt_embeds = self.gpt2.transformer.wte(input_ids)
            start_embeds = torch.cat([prefix_embeds, promt_embeds], dim=1)

            prefix_mask = torch.ones(
                B, 1 + self.num_img_tokens, device=img_features.device
            )
            start_mask = torch.cat([prefix_mask, attention_mask], dim=1)

        else:
            start_embeds = prefix_embeds
            start_mask = torch.ones(
                B, 1 + self.num_img_tokens, device=img_features.device
            )

        output_ids = self.gpt2.generate(
            inputs_embeds=start_embeds,
            attention_mask=start_mask,
            pad_token_id=self.config.eos_token_id,
            **gen_kwargs,
        )
        return output_ids[:, (self.prefix_len + prompt_len) :]
