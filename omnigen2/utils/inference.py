import argparse
import torch
from accelerate import Accelerator
from diffusers.hooks import apply_group_offloading
from omnigen2.models.transformers.transformer_omnigen2 import OmniGen2Transformer2DModel
from omnigen2.pipelines.omnigen2.pipeline_omnigen2 import OmniGen2Pipeline


def load_pipeline(args: argparse.Namespace, accelerator: Accelerator, weight_dtype: torch.dtype) -> OmniGen2Pipeline:
    pipeline = OmniGen2Pipeline.from_pretrained(
        args.model_path,
        torch_dtype=weight_dtype,
        trust_remote_code=True,
    )

    if args.transformer_path:
        print(f"Transformer weights loaded from {args.transformer_path}")
        pipeline.transformer = OmniGen2Transformer2DModel.from_pretrained(
            args.transformer_path,
            torch_dtype=weight_dtype,
        )
    else:
        pipeline.transformer = OmniGen2Transformer2DModel.from_pretrained(
            args.model_path,
            subfolder="transformer",
            torch_dtype=weight_dtype,
        )

    if args.transformer_lora_path:
        print(f"LoRA weights loaded from {args.transformer_lora_path}")
        pipeline.load_lora_weights(args.transformer_lora_path)

    if args.enable_teacache and args.enable_taylorseer:
        print("WARNING: enable_teacache and enable_taylorseer are mutually exclusive. enable_teacache will be ignored.")

    if args.enable_taylorseer:
        pipeline.enable_taylorseer = True
    elif args.enable_teacache:
        pipeline.transformer.enable_teacache = True
        pipeline.transformer.teacache_rel_l1_thresh = args.teacache_rel_l1_thresh

    if args.scheduler == "dpmsolver++":
        from omnigen2.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler
        scheduler = DPMSolverMultistepScheduler(
            algorithm_type="dpmsolver++",
            solver_type="midpoint",
            solver_order=2,
            prediction_type="flow_prediction",
        )
        pipeline.scheduler = scheduler

    if args.enable_sequential_cpu_offload:
        pipeline.enable_sequential_cpu_offload()
    elif args.enable_model_cpu_offload:
        pipeline.enable_model_cpu_offload()
    elif args.enable_group_offload:
        apply_group_offloading(pipeline.transformer, onload_device=accelerator.device, offload_type="block_level", num_blocks_per_group=2, use_stream=True)
        apply_group_offloading(pipeline.mllm, onload_device=accelerator.device, offload_type="block_level", num_blocks_per_group=2, use_stream=True)
        apply_group_offloading(pipeline.vae, onload_device=accelerator.device, offload_type="block_level", num_blocks_per_group=2, use_stream=True)
    else:
        pipeline = pipeline.to(accelerator.device)

    return pipeline