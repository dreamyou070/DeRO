import math
import inspect
import numpy as np
from typing import Any, Dict, Optional, Tuple, Union, List, Callable
import torch
from transformers import AutoProcessor
import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPTokenizer
from torch import nn
import torch
from typing import Optional
from torch import nn
import torch
from diffusers import FluxPipeline
from torchvision.transforms import ToPILImage
import torch
from einops import rearrange

from diffusers.models.attention import _chunked_feed_forward
from diffusers.models.unets.unet_2d_condition import UNet2DConditionOutput
from diffusers.models.transformers.transformer_2d import Transformer2DModelOutput
from diffusers.pipelines.flux.pipeline_flux import (
    retrieve_timesteps,
    replace_example_docstring,
    EXAMPLE_DOC_STRING,
    calculate_shift,
    XLA_AVAILABLE,
    FluxPipelineOutput
)
from diffusers.utils import (
    deprecate,
    BaseOutput,
    is_torch_version,
    logging,
    USE_PEFT_BACKEND,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.models.attention_processor import (
    Attention,
    AttnProcessor,
    AttnProcessor2_0,
)
import torch.nn.functional as F

logger = logging.get_logger(__name__)

attn_maps = {}

@torch.no_grad()
def FluxPipeline_call(self,
                      prompt: Union[str, List[str]] = None,
                      others: Union[str, List[str]] = None,
                      prompt_2: Optional[Union[str, List[str]]] = None,
                      height: Optional[int] = None,
                      width: Optional[int] = None,
                      num_inference_steps: int = 28,
                      timesteps: List[int] = None,
                      guidance_scale: float = 3.5,
                      negation_token="",
                      iterative = False,
                      base_entity="",
                      base_entity_biased_prompts: Optional[Union[str, List[str]]] = None,
                      base_entity_bias_absent_prompts: Optional[Union[str, List[str]]] = None,
                      component_entity = "",
                      extraction_method = 'svd',
                      extraction_k=1,
                      mode = '',
                      base=False,
                      style_ratio=0.25,

                      num_images_per_prompt: Optional[int] = 1,
                      generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
                      latents: Optional[torch.FloatTensor] = None,
                      prompt_embeds: Optional[torch.FloatTensor] = None,
                      pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
                      output_type: Optional[str] = "pil",
                      return_dict: bool = True,
                      joint_attention_kwargs: Optional[Dict[str, Any]] = None,
                      callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
                      callback_on_step_end_tensor_inputs: List[str] = ["latents"],
                      max_sequence_length: int = 512):

    height = height or self.default_sample_size * self.vae_scale_factor
    width = width or self.default_sample_size * self.vae_scale_factor

    # 1. Check inputs. Raise error if not correct
    self.check_inputs(prompt,prompt_2,height,width,prompt_embeds=prompt_embeds,pooled_prompt_embeds=pooled_prompt_embeds,callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,max_sequence_length=max_sequence_length)
    self._guidance_scale = guidance_scale
    self._joint_attention_kwargs = joint_attention_kwargs
    self._interrupt = False

    if prompt is not None and isinstance(prompt, str): batch_size = 1
    elif prompt is not None and isinstance(prompt, list): batch_size = len(prompt)
    else: batch_size = prompt_embeds.shape[0]
    device = self._execution_device
    lora_scale = (self.joint_attention_kwargs.get("scale", None) if self.joint_attention_kwargs is not None else None)

    # ====================================================================================================================================================== #
    def _get_t5_prompt_embeds(prompt: Union[str, List[str]] = None,
                              num_images_per_prompt: int = 1,max_sequence_length: int = 512,device: Optional[torch.device] = None,
                              dtype: Optional[torch.dtype] = None,object_word=None,target_word=None):
        device = device or self._execution_device
        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)
        text_inputs = self.tokenizer_2(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt")
        text_input_ids = text_inputs.input_ids
        t_inputs = self.tokenizer_2(prompt,
                                    padding=False,
                                    truncation=True,
                                    return_length=False,
                                    return_overflowing_tokens=False,
                                    return_tensors="pt")
        input_ids = t_inputs.input_ids[0]
        txt_len = t_inputs.input_ids.shape[-1]
        object_token_indexs = []
        target_token_indexs = []
        if target_word is not None:
            input_ids = input_ids.tolist()
            target_ids = self.tokenizer_2(target_word, padding=False, return_tensors="pt", ).input_ids[0]
            target_ids = target_ids.tolist()
            if target_ids[0] == 3:
                target_ids = target_ids[1:-1]
            else:
                target_ids = target_ids[0:-1]
            for target_idx in target_ids:
                for j, sen_idx in enumerate(input_ids):
                    if target_idx == sen_idx:
                        if j not in target_token_indexs:
                            target_token_indexs.append(j)

            object_ids = self.tokenizer_2(object_word, padding=False, return_tensors="pt", ).input_ids[0]
            object_ids = object_ids.tolist()
            if object_ids[0] == 3:
                object_ids = object_ids[1:-1]
            else:
                object_ids = object_ids[0:-1]
            for object_idx in object_ids:
                for jj, sen_idx in enumerate(input_ids):
                    if object_idx == sen_idx:
                        if jj not in object_token_indexs:
                            object_token_indexs.append(jj)
        untruncated_ids = self.tokenizer_2(prompt, padding="longest", return_tensors="pt").input_ids
        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer_2.batch_decode(untruncated_ids[:, self.tokenizer_max_length - 1: -1])
            logger.warning("The following part of your input was truncated because `max_sequence_length` is set to "
                           f" {max_sequence_length} tokens: {removed_text}")
        prompt_embeds = self.text_encoder_2(text_input_ids.to(device), output_hidden_states=False)[0]
        dtype = self.text_encoder_2.dtype
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)
        return prompt_embeds, txt_len, object_token_indexs, target_token_indexs
    def encode_prompt(prompt: Union[str, List[str]],prompt_2: Union[str, List[str]],device: Optional[torch.device] = None,num_images_per_prompt: int = 1,prompt_embeds: Optional[torch.FloatTensor] = None,pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
                      max_sequence_length: int = 512,lora_scale: Optional[float] = None,base_entity=None,negation_token=None,):
        device = device or self._execution_device
        if lora_scale is not None:
            self._lora_scale = lora_scale
            if self.text_encoder is not None and USE_PEFT_BACKEND:
                scale_lora_layers(self.text_encoder, lora_scale)
            if self.text_encoder_2 is not None and USE_PEFT_BACKEND:
                scale_lora_layers(self.text_encoder_2, lora_scale)

        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt_embeds is None:
            prompt_2 = prompt_2 or prompt
            prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2
            pooled_prompt_embeds = self._get_clip_prompt_embeds(
                prompt=prompt,device=device,
                num_images_per_prompt=num_images_per_prompt)
            prompt_embeds, txt_len, object_token_ids, target_token_ids = _get_t5_prompt_embeds(
                prompt=prompt_2,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                object_word=base_entity,
                target_word=negation_token)
        dtype = self.text_encoder.dtype if self.text_encoder is not None else self.transformer.dtype
        text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(device=device, dtype=dtype)
        return prompt_embeds, pooled_prompt_embeds, text_ids, txt_len, object_token_ids, target_token_ids

    # [1]
    (prompt_embeds, pooled_prompt_embeds,text_ids, txt_len, object_token_ids, component_entity_ids) = encode_prompt(prompt=prompt,
                                                                                                                    prompt_2=prompt_2,
                                                                                                                    prompt_embeds=prompt_embeds,
                                                                                                                    pooled_prompt_embeds=pooled_prompt_embeds,
                                                                                                                    device=device,
                                                                                                                    num_images_per_prompt=num_images_per_prompt,
                                                                                                                    max_sequence_length=max_sequence_length,
                                                                                                                    lora_scale=lora_scale,
                                                                                                                    base_entity=base_entity,
                                                                                                                    negation_token=component_entity)
    prompt_embeds_org = prompt_embeds[0].unsqueeze(dim=0)
    prompt_embeds_edit = prompt_embeds[1].unsqueeze(dim=0)
    pooled_prompt_embeds_org = pooled_prompt_embeds[0].unsqueeze(dim=0)
    pooled_prompt_embeds_edit = pooled_prompt_embeds[1].unsqueeze(dim=0)

    base_entity_embedding       = prompt_embeds_edit[:,object_token_ids,:] # bakery embedding
    component_entity_embedding  = prompt_embeds_edit[:,component_entity_ids,:] # bread embedding

    ##########################################################################################################################################################
    # [2] base_entity_biased_prompts
    chunk_size = 10
    base_entity_biased_prompt_embeds = []
    base_entity_biased_prompt_pooled_embeds = []
    all_text_ids = []
    for i in range(0, len(base_entity_biased_prompts), chunk_size):
        base_entity_biased_prompt_chunk = base_entity_biased_prompts[i:i + chunk_size]
        (chunk_prompt_embeds, chunk_pooled_prompt_embeds, chunk_text_ids,) = self.encode_prompt(prompt=base_entity_biased_prompt_chunk,
                                                                                                prompt_2=base_entity_biased_prompt_chunk,
                                                                                                prompt_embeds=None,pooled_prompt_embeds=None,
                                                                                                device=device,num_images_per_prompt=num_images_per_prompt,
                                                                                                max_sequence_length=max_sequence_length,lora_scale=lora_scale)
        base_entity_biased_prompt_embeds.append(chunk_prompt_embeds)
        base_entity_biased_prompt_pooled_embeds.append(chunk_pooled_prompt_embeds)
    base_entity_biased_prompt_embeds = torch.cat(base_entity_biased_prompt_embeds, dim=0)
    base_entity_biased_prompt_pooled_embeds = torch.cat(base_entity_biased_prompt_pooled_embeds, dim=0)

    # [3] base_entity_biased_prompts
    chunk_size = 10
    base_entity_bias_absent_prompts_embeds = []
    base_entity_bias_absent_prompts_pooled_embeds = []
    all_text_ids = []
    for i in range(0, len(base_entity_bias_absent_prompts), chunk_size):
        base_entity_biased_absent_prompt_chunk = base_entity_bias_absent_prompts[i:i + chunk_size]
        (chunk_prompt_embeds, chunk_pooled_prompt_embeds, chunk_text_ids,) = self.encode_prompt(
            prompt=base_entity_biased_absent_prompt_chunk,
            prompt_2=base_entity_biased_absent_prompt_chunk,
            prompt_embeds=None, pooled_prompt_embeds=None,
            device=device, num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length, lora_scale=lora_scale)
        base_entity_bias_absent_prompts_embeds.append(chunk_prompt_embeds)
        base_entity_bias_absent_prompts_pooled_embeds.append(chunk_pooled_prompt_embeds)
    base_entity_bias_absent_prompts_embeds = torch.cat(base_entity_bias_absent_prompts_embeds, dim=0)
    base_entity_bias_absent_prompts_pooled_embeds = torch.cat(base_entity_bias_absent_prompts_pooled_embeds, dim=0)
    ########################################################################################################################
    # [4] get absent pooled space
    def extract_common_and_inject_object_tokens(prompt_embeds: torch.Tensor,
                                                pooled_prompt_embeds: torch.Tensor,  # (2, dim_g)
                                                other_prompt_embeds: torch.Tensor,  # (B, seq, dim)  (reference prompts)
                                                other_pooled_prompt_embeds: torch.Tensor,  # (B, dim_g)
                                                object_token_ids: torch.Tensor | list[int],  # token indices to replace
                                                extraction_method: str = "pca",
                                                # "svd" | "pca" | "rpca" | "ica" | "none"
                                                k: int | None = None,
                                                energy: float = 0.95,
                                                iterative=False):
        def anal_svd(a_vec: torch.Tensor, b_vec: torch.Tensor, k: int | None = 1, energy: float = 0.95,
                     eps: float = 1e-12, ):
            #X = torch.cat([a_vec, b_vec], dim=0).float()  # (N, D)
            X = b_vec
            mu = X.mean(dim=0, keepdim=True)  # (1, D)
            Xc = X - mu  # (N, D)
            U, S, Vh = torch.linalg.svd(Xc, full_matrices=False)  # Vh: (r, D)
            k_eff = int(k)
            k_eff = max(1, min(k_eff, Vh.shape[0]))
            B = Vh[:k_eff, :].T.contiguous()  # (D, k)
            a_c = a_vec.float() - mu  # (1, D)
            a_proj = (a_c @ B) @ B.T  # (1, D)
            a_ortho = a_c - a_proj  # (1, D)
            return mu, B, S, k_eff, a_proj, a_ortho
        prompt_embeds_org = prompt_embeds
        a_vec = prompt_embeds.clone().reshape(1, -1)  # 512,1024
        a_pool = pooled_prompt_embeds.clone().reshape(1, -1)  # 1,768
        if a_pool.dim() != 2: a_pool = a_pool.unsqueeze(dim=0)
        b = other_prompt_embeds.shape[0]
        b_vec = other_prompt_embeds.reshape(b, -1)  # batch,1024
        b_pool = other_pooled_prompt_embeds  # batch,769
        mu_l, B_l, S_l, k_l, proj_l, ortho_l = anal_svd(a_vec, b_vec, k=(1 if k is None else k), energy=energy)
        mu_g, B_g, S_g, k_g, proj_g, ortho_g = anal_svd(a_pool, b_pool, k=(1 if k is None else k), energy=energy)
        prompt_embeds_edit = (mu_l + proj_l).reshape(1, 512, -1).to(prompt_embeds_org.dtype)
        pooled_edit = (mu_g + proj_g)
        if pooled_edit.dim() != 2:
            pooled_edit = pooled_edit.unsqueeze(dim=0)
        if prompt_embeds_edit.dim() != 3 :
            prompt_embeds_edit = prompt_embeds_edit.unsqueeze(dim=0)
        return prompt_embeds_edit, pooled_edit

    # One Steps SVD
    prompt_embeds_edit = prompt_embeds_edit.to(dtype = torch.float32)
    pooled_prompt_embeds_edit = pooled_prompt_embeds_edit.to(dtype = torch.float32)
    base_entity_bias_absent_prompts_embeds = base_entity_bias_absent_prompts_embeds.to(dtype = torch.float32)
    base_entity_bias_absent_prompts_pooled_embeds = base_entity_bias_absent_prompts_pooled_embeds.to(dtype = torch.float32)
    prompt_embeds_edit_proj, pooled_prompt_embeds_edit = extract_common_and_inject_object_tokens(prompt_embeds=prompt_embeds_edit,
                                                                                                 pooled_prompt_embeds=pooled_prompt_embeds_edit,
                                                                                                 other_prompt_embeds= base_entity_bias_absent_prompts_embeds,

                                                                                                 other_pooled_prompt_embeds=base_entity_bias_absent_prompts_pooled_embeds,
                                                                                                 object_token_ids=object_token_ids,
                                                                                                 extraction_method=extraction_method,
                                                                                                 k=extraction_k,
                                                                                                 energy=0.95,iterative=iterative)

    def extract_common_and_inject_object_tokens_iterative(
            prompt_embeds: torch.Tensor,
            biased_prompt_embeds: torch.Tensor,
            unbiased_prompt_embeds: torch.Tensor,
            k_semantic: int = 1,
            k_bias: int = 1,
            style_ratio: float = 1.0):
        """
        [1] unbiased prompts로부터 공통 semantic subspace 추출
        [2] prompt_embeds를 semantic + unique로 분해
        [3] unique에서 biased prompts 방향과 겹치는 bias 성분 제거해서 style_component 남기기
        [4] semantic + style_component를 다시 합침
        """
        dtype = prompt_embeds.dtype
        device = prompt_embeds.device
        original_shape = prompt_embeds.shape  # usually (1, 512, dim)
        # concat
        # biased_prompt_embeds = torch.cat([prompt_embeds,biased_prompt_embeds], dim=0)

        a_vec = prompt_embeds.reshape(1, -1).float()
        c_vec = biased_prompt_embeds.reshape(biased_prompt_embeds.shape[0], -1).float()
        c_vec = c_vec.to(device)
        def project_to_svd_subspace(target_vec, ref_vecs, k=1):
            """
            target_vec: (1, D)
            ref_vecs: (B, D)
            return:
                projected_component: target가 ref subspace에 투영된 성분
                residual_component: target에서 projected_component를 뺀 성분
            """
            #
            X = ref_vecs
            # reference prompts만으로 subspace 구성
            mu = X.mean(dim=0, keepdim=True)
            Xc = X - mu
            _, _, Vh = torch.linalg.svd(Xc, full_matrices=False)
            k_eff = max(1, min(int(k), Vh.shape[0]))
            B = Vh[:k_eff, :].T.contiguous()  # (D, k)
            target_centered = target_vec - mu
            projected_component = (target_centered @ B) @ B.T
            residual_component = target_centered - projected_component
            projected_component = projected_component + mu
            return projected_component, residual_component
        bias_component, style_component = project_to_svd_subspace(target_vec=a_vec,
                                                                  ref_vecs=c_vec,
                                                                  k=k_bias,)
        final_vec = a_vec - bias_component #+ style_ratio * style_component
        final = final_vec.reshape(original_shape).to(dtype)
        return final

    prompt_embeds_edit = extract_common_and_inject_object_tokens_iterative(
        prompt_embeds=prompt_embeds_edit,
        k_bias=extraction_k,
        biased_prompt_embeds = base_entity_biased_prompt_embeds,
        unbiased_prompt_embeds = base_entity_bias_absent_prompts_embeds,
        style_ratio=style_ratio)

    pooled_prompt_embeds = torch.cat([pooled_prompt_embeds_org, pooled_prompt_embeds_edit],dim=0).to(dtype = torch.float16)
    prompt_embeds = torch.cat([prompt_embeds_org, prompt_embeds_edit], dim=0).to(dtype = torch.float16)

    num_channels_latents = self.transformer.config.in_channels // 4
    latents, latent_image_ids = self.prepare_latents(batch_size * num_images_per_prompt,
        num_channels_latents,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
        latents)
    latents_org = latents[0]
    latents_edit = latents[1]
    latents[1] = latents_org
    # 5. Prepare timesteps
    sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
    image_seq_len = latents.shape[1]
    mu = calculate_shift(
        image_seq_len,
        self.scheduler.config.base_image_seq_len,
        self.scheduler.config.max_image_seq_len,
        self.scheduler.config.base_shift,
        self.scheduler.config.max_shift,
    )
    timesteps, num_inference_steps = retrieve_timesteps(
        self.scheduler,
        num_inference_steps,
        device,
        timesteps,
        sigmas,
        mu=mu,
    )
    num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
    self._num_timesteps = len(timesteps)

    # handle guidance
    if self.transformer.config.guidance_embeds:
        guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
        guidance = guidance.expand(latents.shape[0])
    else:
        guidance = None

    # 6. Denoising loop

    bf16 = torch.bfloat16
    # bf16 = torch.float16
    latents = latents.to(dtype=bf16)
    self.transformer = self.transformer.to(dtype=bf16)
    prompt_embeds = prompt_embeds.to(dtype=bf16)
    pooled_prompt_embeds = pooled_prompt_embeds.to(dtype=bf16)
    self.vae = self.vae.to(dtype=bf16)

    latents_init = latents.detach().clone()
    initial_noise = latents_init[1].unsqueeze(dim=0)

    latents = latents_init.detach().clone()
    def decode_preview_from_noise_pred(noise_pred):
        noise_pred_edit = noise_pred[1].unsqueeze(dim=0)

        pred_edit = initial_noise - noise_pred_edit

        pred_edit = self._unpack_latents(
            pred_edit,
            height,
            width,
            self.vae_scale_factor,
        )

        pred_edit = (
                            pred_edit / self.vae.config.scaling_factor
                    ) + self.vae.config.shift_factor

        pred_image = self.vae.decode(pred_edit, return_dict=False)[0]

        pred_image = self.image_processor.postprocess(
            pred_image,
            output_type=output_type,
        )[0]

        return pred_image

    with self.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            if self.interrupt:
                continue

            timestep = t.expand(latents.shape[0]).to(latents.dtype)

            noise_pred = self.transformer(
                hidden_states=latents,
                timestep=timestep / 1000,
                guidance=guidance,
                pooled_projections=pooled_prompt_embeds,
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids,
                img_ids=latent_image_ids,
                joint_attention_kwargs=self.joint_attention_kwargs,
                return_dict=False,
            )[0]

            latents_dtype = latents.dtype

            latents = self.scheduler.step(
                noise_pred,
                t,
                latents,
                return_dict=False,
            )[0]

            if latents.dtype != latents_dtype:
                if torch.backends.mps.is_available():
                    latents = latents.to(latents_dtype)

            pred_image = decode_preview_from_noise_pred(noise_pred)

            pred_image.save(f"pred_image_{i}.png")

            if callback_on_step_end is not None:
                callback_kwargs = {}

                for k in callback_on_step_end_tensor_inputs:
                    callback_kwargs[k] = locals()[k]

                callback_outputs = callback_on_step_end(
                    self,
                    i,
                    t,
                    callback_kwargs,
                )

                latents = callback_outputs.pop("latents", latents)
                prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

            if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps
                    and (i + 1) % self.scheduler.order == 0
            ):
                progress_bar.update()

            if XLA_AVAILABLE:
                xm.mark_step()

    if output_type == "latent":
        image = latents
    else:
        latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
        latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
        image = self.vae.decode(latents, return_dict=False)[0]
        image = self.image_processor.postprocess(image, output_type=output_type)

    # Offload all models
    self.maybe_free_model_hooks()

    if not return_dict:
        return (image,)

    return FluxPipelineOutput(images=image)


def FluxTransformer2DModelForward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        pooled_projections: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_ids: torch.Tensor = None,
        txt_ids: torch.Tensor = None,
        guidance: torch.Tensor = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_block_samples=None,
        controlnet_single_block_samples=None,
        return_dict: bool = True,
        recursive=False,
        controlnet_blocks_repeat: bool = False,
        ##################################################
        height: int = None,
        width: int = None,
        ##################################################
) -> Union[torch.FloatTensor, Transformer2DModelOutput]:
    """
    The [`FluxTransformer2DModel`] forward method.

    Args:
        hidden_states (`torch.FloatTensor` of shape `(batch size, channel, height, width)`):
            Input `hidden_states`.
        encoder_hidden_states (`torch.FloatTensor` of shape `(batch size, sequence_len, embed_dims)`):
            Conditional embeddings (embeddings computed from the input conditions such as prompts) to use.
        pooled_projections (`torch.FloatTensor` of shape `(batch_size, projection_dim)`): Embeddings projected
            from the embeddings of input conditions.
        timestep ( `torch.LongTensor`):
            Used to indicate denoising step.
        block_controlnet_hidden_states: (`list` of `torch.Tensor`):
            A list of tensors that if specified are added to the residuals of transformer blocks.
        joint_attention_kwargs (`dict`, *optional*):
            A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
            `self.processor` in
            [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
        return_dict (`bool`, *optional*, defaults to `True`):
            Whether or not to return a [`~models.transformer_2d.Transformer2DModelOutput`] instead of a plain
            tuple.

    Returns:
        If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
        `tuple` where the first element is the sample tensor.
    """
    if joint_attention_kwargs is not None:
        joint_attention_kwargs = joint_attention_kwargs.copy()
        lora_scale = joint_attention_kwargs.pop("scale", 1.0)
    else:
        lora_scale = 1.0

    if USE_PEFT_BACKEND:
        # weight the lora layers by setting `lora_scale` for each PEFT layer
        scale_lora_layers(self, lora_scale)
    else:
        if joint_attention_kwargs is not None and joint_attention_kwargs.get("scale", None) is not None:
            logger.warning(
                "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
            )
    hidden_states = self.x_embedder(hidden_states)

    timestep = timestep.to(hidden_states.dtype) * 1000
    if guidance is not None:
        guidance = guidance.to(hidden_states.dtype) * 1000
    else:
        guidance = None
    temb = (
        self.time_text_embed(timestep, pooled_projections)
        if guidance is None
        else self.time_text_embed(timestep, guidance, pooled_projections)
    )
    encoder_hidden_states = self.context_embedder(encoder_hidden_states)

    if txt_ids.ndim == 3:
        logger.warning(
            "Passing `txt_ids` 3d torch.Tensor is deprecated."
            "Please remove the batch dimension and pass it as a 2d torch Tensor"
        )
        txt_ids = txt_ids[0]
    if img_ids.ndim == 3:
        logger.warning(
            "Passing `img_ids` 3d torch.Tensor is deprecated."
            "Please remove the batch dimension and pass it as a 2d torch Tensor"
        )
        img_ids = img_ids[0]

    ids = torch.cat((txt_ids, img_ids), dim=0)
    image_rotary_emb = self.pos_embed(ids)

    for index_block, block in enumerate(self.transformer_blocks):
        if torch.is_grad_enabled() and self.gradient_checkpointing:

            def create_custom_forward(module, return_dict=None):
                def custom_forward(*inputs):
                    if return_dict is not None:
                        return module(*inputs, return_dict=return_dict)
                    else:
                        return module(*inputs)

                return custom_forward

            ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
            encoder_hidden_states, hidden_states = torch.utils.checkpoint.checkpoint(
                create_custom_forward(block),
                hidden_states,
                encoder_hidden_states,
                temb,
                image_rotary_emb,
                **ckpt_kwargs,
            )

        else:
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=joint_attention_kwargs,
            )

        # controlnet residual
        if controlnet_block_samples is not None:
            interval_control = len(self.transformer_blocks) / len(controlnet_block_samples)
            interval_control = int(np.ceil(interval_control))
            # For Xlabs ControlNet.
            if controlnet_blocks_repeat:
                hidden_states = (
                        hidden_states + controlnet_block_samples[index_block % len(controlnet_block_samples)]
                )
            else:
                hidden_states = hidden_states + controlnet_block_samples[index_block // interval_control]

    hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

    for index_block, block in enumerate(self.single_transformer_blocks):
        if torch.is_grad_enabled() and self.gradient_checkpointing:

            def create_custom_forward(module, return_dict=None):
                def custom_forward(*inputs):
                    if return_dict is not None:
                        return module(*inputs, return_dict=return_dict)
                    else:
                        return module(*inputs)

                return custom_forward

            ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
            hidden_states = torch.utils.checkpoint.checkpoint(
                create_custom_forward(block),
                hidden_states,
                temb,
                image_rotary_emb,
                **ckpt_kwargs,
            )

        else:
            hidden_states = block(
                hidden_states=hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=joint_attention_kwargs,
            )

        # controlnet residual
        if controlnet_single_block_samples is not None:
            interval_control = len(self.single_transformer_blocks) / len(controlnet_single_block_samples)
            interval_control = int(np.ceil(interval_control))
            hidden_states[:, encoder_hidden_states.shape[1]:, ...] = (
                    hidden_states[:, encoder_hidden_states.shape[1]:, ...]
                    + controlnet_single_block_samples[index_block // interval_control]
            )

    hidden_states = hidden_states[:, encoder_hidden_states.shape[1]:, ...]

    hidden_states = self.norm_out(hidden_states, temb)
    output = self.proj_out(hidden_states)

    if USE_PEFT_BACKEND:
        # remove `lora_scale` from each PEFT layer
        unscale_lora_layers(self, lora_scale)

    if not return_dict:
        return (output,)

    return Transformer2DModelOutput(sample=output)


def attn_call(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        height: int = None,
        width: int = None,
        timestep: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
) -> torch.Tensor:
    if len(args) > 0 or kwargs.get("scale", None) is not None:
        deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
        deprecate("scale", "1.0.0", deprecation_message)

    residual = hidden_states

    if attn.spatial_norm is not None:
        hidden_states = attn.spatial_norm(hidden_states, temb)

    input_ndim = hidden_states.ndim

    if input_ndim == 4:
        batch_size, channel, height, width = hidden_states.shape
        hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

    batch_size, sequence_length, _ = (
        hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
    )
    attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

    if attn.group_norm is not None:
        hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

    query = attn.to_q(hidden_states)

    if encoder_hidden_states is None:
        encoder_hidden_states = hidden_states
    elif attn.norm_cross:
        encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

    key = attn.to_k(encoder_hidden_states)
    value = attn.to_v(encoder_hidden_states)

    query = attn.head_to_batch_dim(query)
    key = attn.head_to_batch_dim(key)
    value = attn.head_to_batch_dim(value)

    attention_probs = attn.get_attention_scores(query, key, attention_mask)
    ####################################################################################################
    if hasattr(self, "store_attn_map"):
        self.attn_map = rearrange(attention_probs, 'b (h w) d -> b d h w', h=height)
        self.timestep = int(timestep.item())
    ####################################################################################################
    hidden_states = torch.bmm(attention_probs, value)
    hidden_states = attn.batch_to_head_dim(hidden_states)

    # linear proj
    hidden_states = attn.to_out[0](hidden_states)
    # dropout
    hidden_states = attn.to_out[1](hidden_states)

    if input_ndim == 4:
        hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

    if attn.residual_connection:
        hidden_states = hidden_states + residual

    hidden_states = hidden_states / attn.rescale_output_factor

    return hidden_states


def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False,
                                 scale=None) -> torch.Tensor:
    # Efficient implementation equivalent to the following:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_mask.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias += attn_mask
    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias.to(attn_weight.device)
    attn_weight = torch.softmax(attn_weight, dim=-1)

    return torch.dropout(attn_weight, dropout_p, train=True) @ value, attn_weight


def attn_call2_0(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        height: int = None,
        width: int = None,
        timestep: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
) -> torch.Tensor:
    if len(args) > 0 or kwargs.get("scale", None) is not None:
        deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
        deprecate("scale", "1.0.0", deprecation_message)

    residual = hidden_states
    if attn.spatial_norm is not None:
        hidden_states = attn.spatial_norm(hidden_states, temb)

    input_ndim = hidden_states.ndim

    if input_ndim == 4:
        batch_size, channel, height, width = hidden_states.shape
        hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

    batch_size, sequence_length, _ = (
        hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
    )

    if attention_mask is not None:
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        # scaled_dot_product_attention expects attention_mask shape to be
        # (batch, heads, source_length, target_length)
        attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

    if attn.group_norm is not None:
        hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

    query = attn.to_q(hidden_states)

    if encoder_hidden_states is None:
        encoder_hidden_states = hidden_states
    elif attn.norm_cross:
        encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

    key = attn.to_k(encoder_hidden_states)
    value = attn.to_v(encoder_hidden_states)

    inner_dim = key.shape[-1]
    head_dim = inner_dim // attn.heads

    query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

    key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
    value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

    # the output of sdp = (batch, num_heads, seq_len, head_dim)
    # TODO: add support for attn.scale when we move to Torch 2.1
    ####################################################################################################
    if hasattr(self, "store_attn_map"):
        hidden_states, attention_probs = scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        self.attn_map = rearrange(
            attention_probs,
            'batch attn_head (h w) attn_dim -> batch attn_head h w attn_dim ',
            h=height
        )  # detach height*width
        self.timestep = int(timestep.item())
    else:
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
    ####################################################################################################

    hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1,
                                                          attn.heads * head_dim)  # (b,attn_head,h*w,attn_dim) -> (b,h*w,attn_head*attn_dim)
    hidden_states = hidden_states.to(query.dtype)

    # linear proj
    hidden_states = attn.to_out[0](hidden_states)
    # dropout
    hidden_states = attn.to_out[1](hidden_states)

    if input_ndim == 4:
        hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

    if attn.residual_connection:
        hidden_states = hidden_states + residual

    hidden_states = hidden_states / attn.rescale_output_factor

    return hidden_states


def lora_attn_call(self, attn: Attention, hidden_states, height, width, *args, **kwargs):
    self_cls_name = self.__class__.__name__
    deprecate(
        self_cls_name,
        "0.26.0",
        (
            f"Make sure use {self_cls_name[4:]} instead by setting"
            "LoRA layers to `self.{to_q,to_k,to_v,to_out[0]}.lora_layer` respectively. This will be done automatically when using"
            " `LoraLoaderMixin.load_lora_weights`"
        ),
    )
    attn.to_q.lora_layer = self.to_q_lora.to(hidden_states.device)
    attn.to_k.lora_layer = self.to_k_lora.to(hidden_states.device)
    attn.to_v.lora_layer = self.to_v_lora.to(hidden_states.device)
    attn.to_out[0].lora_layer = self.to_out_lora.to(hidden_states.device)

    attn._modules.pop("processor")
    attn.processor = AttnProcessor()
    ####################################################################################################
    attn.processor.__call__ = attn_call.__get__(attn.processor, AttnProcessor)
    ####################################################################################################

    if hasattr(self, "store_attn_map"):
        attn.processor.store_attn_map = True

    return attn.processor(attn, hidden_states, height, width, *args, **kwargs)


def lora_attn_call2_0(self, attn: Attention, hidden_states, height, width, *args, **kwargs):
    self_cls_name = self.__class__.__name__
    deprecate(
        self_cls_name,
        "0.26.0",
        (
            f"Make sure use {self_cls_name[4:]} instead by setting"
            "LoRA layers to `self.{to_q,to_k,to_v,to_out[0]}.lora_layer` respectively. This will be done automatically when using"
            " `LoraLoaderMixin.load_lora_weights`"
        ),
    )
    attn.to_q.lora_layer = self.to_q_lora.to(hidden_states.device)
    attn.to_k.lora_layer = self.to_k_lora.to(hidden_states.device)
    attn.to_v.lora_layer = self.to_v_lora.to(hidden_states.device)
    attn.to_out[0].lora_layer = self.to_out_lora.to(hidden_states.device)

    attn._modules.pop("processor")
    attn.processor = AttnProcessor2_0()
    ####################################################################################################
    attn.processor.__call__ = attn_call.__get__(attn.processor, AttnProcessor2_0)
    ####################################################################################################
    if hasattr(self, "store_attn_map"):
        attn.processor.store_attn_map = True

    return attn.processor(attn, hidden_states, height, width, *args, **kwargs)


def joint_attn_call2_0(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        ############################################################
        height: int = None,
        timestep: Optional[torch.Tensor] = None,
        ############################################################
        *args,
        **kwargs,
) -> torch.FloatTensor:
    residual = hidden_states

    batch_size = hidden_states.shape[0]

    # `sample` projections.
    query = attn.to_q(hidden_states)
    key = attn.to_k(hidden_states)
    value = attn.to_v(hidden_states)

    inner_dim = key.shape[-1]
    head_dim = inner_dim // attn.heads

    query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
    key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
    value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

    if attn.norm_q is not None:
        query = attn.norm_q(query)
    if attn.norm_k is not None:
        key = attn.norm_k(key)

    # `context` projections.
    if encoder_hidden_states is not None:
        encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
        encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
        encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

        encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
            batch_size, -1, attn.heads, head_dim
        ).transpose(1, 2)
        encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
            batch_size, -1, attn.heads, head_dim
        ).transpose(1, 2)
        encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
            batch_size, -1, attn.heads, head_dim
        ).transpose(1, 2)

        if attn.norm_added_q is not None:
            encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
        if attn.norm_added_k is not None:
            encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

        query = torch.cat([query, encoder_hidden_states_query_proj], dim=2)
        key = torch.cat([key, encoder_hidden_states_key_proj], dim=2)
        value = torch.cat([value, encoder_hidden_states_value_proj], dim=2)

    ####################################################################################################
    if hasattr(self, "store_attn_map") and encoder_hidden_states is not None:
        hidden_states, attention_probs = scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        image_length = query.shape[2] - encoder_hidden_states_query_proj.shape[2]

        # (4,24,4429,4429) -> (4,24,4096,333)
        attention_probs = attention_probs[:, :, :image_length, image_length:].cpu()

        self.attn_map = rearrange(
            attention_probs,
            'batch attn_head (height width) attn_dim -> batch attn_head height width attn_dim',
            height=height
        )  # (4, 24, 4096, 333) -> (4, 24, height, width, 333)
        self.timestep = timestep[0].cpu().item()  # TODO: int -> list
    else:
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
    ####################################################################################################

    # hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
    hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
    hidden_states = hidden_states.to(query.dtype)

    if encoder_hidden_states is not None:
        # Split the attention outputs.
        hidden_states, encoder_hidden_states = (
            hidden_states[:, : residual.shape[1]],
            hidden_states[:, residual.shape[1]:],
        )
        if not attn.context_pre_only:
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

    # linear proj
    hidden_states = attn.to_out[0](hidden_states)
    # dropout
    hidden_states = attn.to_out[1](hidden_states)

    if encoder_hidden_states is not None:
        return hidden_states, encoder_hidden_states
    else:
        return hidden_states


def flux_attn_call2_0(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
) -> torch.FloatTensor:
    batch_size, _, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape

    # `sample` projections.
    query = attn.to_q(hidden_states)
    key = attn.to_k(hidden_states)
    value = attn.to_v(hidden_states)

    inner_dim = key.shape[-1]
    head_dim = inner_dim // attn.heads

    query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
    key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
    value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

    if attn.norm_q is not None:
        query = attn.norm_q(query)
    if attn.norm_k is not None:
        key = attn.norm_k(key)

    # the attention in FluxSingleTransformerBlock does not use `encoder_hidden_states`
    if encoder_hidden_states is not None:
        # `context` projections.
        encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
        encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
        encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

        encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
            batch_size, -1, attn.heads, head_dim
        ).transpose(1, 2)
        encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
            batch_size, -1, attn.heads, head_dim
        ).transpose(1, 2)
        encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
            batch_size, -1, attn.heads, head_dim
        ).transpose(1, 2)

        if attn.norm_added_q is not None:
            encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
        if attn.norm_added_k is not None:
            encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

        # attention
        query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
        key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
        value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)

    if image_rotary_emb is not None:
        from diffusers.models.embeddings import apply_rotary_emb

        query = apply_rotary_emb(query, image_rotary_emb)
        key = apply_rotary_emb(key, image_rotary_emb)

    hidden_states = F.scaled_dot_product_attention(
        query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
    )
    ####################################################################################################
    hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
    hidden_states = hidden_states.to(query.dtype)

    if encoder_hidden_states is not None:
        encoder_hidden_states, hidden_states = (
            hidden_states[:, : encoder_hidden_states.shape[1]],
            hidden_states[:, encoder_hidden_states.shape[1]:],
        )

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        return hidden_states, encoder_hidden_states
    else:
        return hidden_states


def FluxSingleTransformerBlockForward(
        self,
        hidden_states: torch.FloatTensor,
        temb: torch.FloatTensor,
        image_rotary_emb=None,
        joint_attention_kwargs=None, ):
    residual = hidden_states
    norm_hidden_states, gate = self.norm(hidden_states, emb=temb)
    mlp_hidden_states = self.act_mlp(self.proj_mlp(norm_hidden_states))
    joint_attention_kwargs = joint_attention_kwargs or {}

    attn_output = self.attn(hidden_states=norm_hidden_states,
                            image_rotary_emb=image_rotary_emb,
                            **joint_attention_kwargs,
                            )

    hidden_states = torch.cat([attn_output, mlp_hidden_states], dim=2)
    gate = gate.unsqueeze(1)
    hidden_states = gate * self.proj_out(hidden_states)
    hidden_states = residual + hidden_states
    if hidden_states.dtype == torch.float16:
        hidden_states = hidden_states.clip(-65504, 65504)

    return hidden_states


def FluxTransformerBlockForward(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor,
        temb: torch.FloatTensor,
        image_rotary_emb=None,
        joint_attention_kwargs=None,
        width: int = None,
):
    norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)
    norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
        encoder_hidden_states, emb=temb
    )
    joint_attention_kwargs = joint_attention_kwargs or {}
    # Attention.
    attn_output, context_attn_output = self.attn(
        hidden_states=norm_hidden_states,
        encoder_hidden_states=norm_encoder_hidden_states,
        image_rotary_emb=image_rotary_emb,
        **joint_attention_kwargs,
    )

    # Process attention outputs for the `hidden_states`.
    attn_output = gate_msa.unsqueeze(1) * attn_output
    hidden_states = hidden_states + attn_output

    norm_hidden_states = self.norm2(hidden_states)
    norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

    ff_output = self.ff(norm_hidden_states)
    ff_output = gate_mlp.unsqueeze(1) * ff_output

    hidden_states = hidden_states + ff_output

    # Process attention outputs for the `encoder_hidden_states`.

    context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
    encoder_hidden_states = encoder_hidden_states + context_attn_output

    norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
    norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]

    context_ff_output = self.ff_context(norm_encoder_hidden_states)
    encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output
    if encoder_hidden_states.dtype == torch.float16:
        encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)

    return encoder_hidden_states, hidden_states

def init_pipeline(pipeline):
    if "transformer" in vars(pipeline) and pipeline.transformer.__class__.__name__ == "FluxTransformer2DModel":
        FluxPipeline.__call__ = FluxPipeline_call
    return pipeline