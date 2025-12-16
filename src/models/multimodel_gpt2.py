import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel


class MultimodelGPT2(nn.Module):
    def __init__(
        self,
        gpt2_model_name="gpt2",
        num_img_tokens=4,
        num_freeze_layers=4,
        dropout_rate=0.1,
    ):
        super().__init__()

        self.gpt2 = GPT2LMHeadModel.from_pretrained(gpt2_model_name)
        self.config = self.gpt2.config
        self.hidden_size = self.config.hidden_size
        self.num_img_tokens = num_img_tokens

        self.image_projection = nn.Sequential(
            nn.Linear(2048, self.hidden_size * self.num_img_tokens),
            nn.LayerNorm(self.hidden_size * self.num_img_tokens),
        )

        self.image_dropout = nn.Dropout(dropout_rate)
        self.learned_bos_embedding = nn.Parameter(torch.randn(1, 1, self.hidden_size))
        self.learned_bos_embedding.requires_grad = True

        for name, param in self.gpt2.named_parameters():
            param.requires_grad = False

        for param in self.image_projection.parameters():
            param.requires_grad = True

        for i in range(self.config.n_layer - num_freeze_layers, self.config.n_layer):
            layer_prefix = f"transformer.h.{i}"
            for name, param in self.gpt2.named_parameters():
                if name.startswith(layer_prefix):
                    param.requires_grad = True

        for name, param in self.gpt2.named_parameters():
            if name.startswith("lm_head"):
                param.requires_grad = True

    def forward(self, img_features, input_ids, attention_mask=None, labels=None):
        B = img_features.shape[0]
        T_text = input_ids.shape[1]
        T_total = T_text + self.num_img_tokens

        projected_features = self.image_projection(img_features)
        image_embeddings = projected_features.view(
            B, self.num_img_tokens, self.hidden_size
        )
        image_embeddings = self.image_dropout(image_embeddings)
        text_embedding = self.gpt2.transformer.wte(input_ids)

        combined_embedding = torch.cat((image_embeddings, text_embedding), dim=1)

        position_ids = (
            torch.arange(T_total, dtype=torch.long, device=input_ids.device)
            .unsqueeze(0)
            .repeat(B, 1)
        )

        if attention_mask is not None:
            images_attention = torch.ones(
                B,
                self.num_img_tokens,
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
            new_attention_mask = torch.cat((images_attention, attention_mask), dim=1)

        else:
            new_attention_mask = None

        new_labels = None

        if labels is not None:
            ignore_token = torch.full(
                (B, self.num_img_tokens), -100, dtype=labels.dtype, device=labels.device
            )
            new_labels = torch.cat((ignore_token, labels), dim=1)

        output = self.gpt2(
            inputs_embeds=combined_embedding,
            attention_mask=new_attention_mask,
            position_ids=position_ids,
            labels=new_labels,
            return_dict=True,
        )

        loss = output.loss

        logits = output.logits[:, self.num_img_tokens :, :]

        if labels is not None:
            return loss, logits

        return logits

    def generate(
        self,
        img_features,
        prompt_text="",
        max_length=50,
        top_k=50,
        top_p=0.9,
        temperature=0.8,
        **kwargs,
    ):

        B = img_features.shape[0]

        projected_features = self.image_projection(img_features)
        image_embeddings = projected_features.view(
            B, self.num_img_tokens, self.hidden_size
        )
        if prompt_text:
            prompt_encoded = self.gpt2.config.tokenizer(
                prompt_text, return_tensors="pt", padding=True
            ).to(img_features.device)
            prompt_ids = prompt_encoded.input_ids.repeat(B, 1)  # [B, T_prompt]
            prompt_mask = prompt_encoded.attention_mask.repeat(B, 1)  # [B, T_prompt]

            prompt_embeddings = self.gpt2.transformer.wte(prompt_ids)

            # Concatenate: [Image Embeddings] + [Prompt Embeddings]
            initial_embeddings = torch.cat((image_embeddings, prompt_embeddings), dim=1)

            # Concatenate Masks: [Image Mask] + [Prompt Mask]
            image_attention = torch.ones(
                B,
                self.num_img_tokens,
                dtype=prompt_mask.dtype,
                device=img_features.device,
            )
            initial_attention_mask = torch.cat((image_attention, prompt_mask), dim=1)

            T_prefix = self.num_img_tokens + prompt_ids.shape[1]  # Total prefix length

        else:
            # Case: Only Image Prefix (Used if no prompt_text is supplied)
            initial_embeddings = image_embeddings
            initial_attention_mask = torch.ones(
                B, self.num_img_tokens, dtype=torch.long, device=img_features.device
            )
            T_prefix = self.num_img_tokens

        # 2. Critical Positional IDs for Generation
        position_ids = (
            torch.arange(T_prefix, dtype=torch.long, device=img_features.device)
            .unsqueeze(0)
            .repeat(B, 1)
        )
        initial_attention_mask = torch.ones(
            B, self.num_img_tokens, dtype=torch.long, device=img_features.device
        )

        output_ids = self.gpt2.generate(
            inputs_embeds=initial_embeddings,
            attention_mask=initial_attention_mask,
            position_ids=position_ids,
            do_sample=True,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            max_length=max_length + self.num_img_tokens,
            pad_token_id=self.gpt2.config.eos_token_id,
            num_return_sequences=1,
            **kwargs,
        )
        return output_ids[:, T_prefix :]
