#!/usr/bin/env python3
"""
Example usage:
    uv run benchmark.py --num_layers 12 --d_model 768 --num_heads 12 --d_ff 3072 --batch_size 4 --seq_len 512
"""

import argparse
import torch
import numpy as np
from timeit import default_timer as timer

from cs336_basics.TransformerLM.transformer_lm import TransformerLM
from cs336_basics.Cross_entropy_loss_AdamW.cross_entropy import run_cross_entropy


def generate_random_batch( batch_size: int, seq_len: int, vocab_size: int = 10000, device: str = 'cuda') -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    y = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    return x, y


def benchmark_step( model: TransformerLM, x: torch.Tensor, y: torch.Tensor, forward_only: bool = False) -> None:
    forward = model(x)
    
    if not forward_only:
        loss = run_cross_entropy(forward, y)
        loss.backward()
        
        model.zero_grad()


def run_benchmark(
    model: TransformerLM,
    x: torch.Tensor,
    y: torch.Tensor,
    warmup_steps: int,
    iter_steps: int,
    forward_only: bool = False,
    device: torch.device = None
) -> tuple[float, float, list[float]]:
    if device is None:
        device = x.device
    
    print(f"Running {warmup_steps} warm-up steps...")
    
    for i in range(warmup_steps):
        benchmark_step(model, x, y, forward_only)
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        print(f"  Warm-up {i+1}/{warmup_steps} complete")
    
    print(f"\nRunning {iter_steps} iteration steps...")
    
    # Iteration phase
    times = []
    for i in range(iter_steps):
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        
        start = timer()
        
        benchmark_step(model, x, y, forward_only)
        
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        
        end = timer()
        
        elapsed = end - start
        times.append(elapsed)
        print(f"  Step {i+1}/{iter_steps}: {elapsed*1000:.2f} ms")
    
    mean_time = np.mean(times)
    std_time = np.std(times)
    
    return mean_time, std_time, times


def get_parser():
    parser = argparse.ArgumentParser(
        description='Benchmark Transformer model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    model_group = parser.add_argument_group('Model Architecture')
    model_group.add_argument('--vocab_size', type=int, default=10000)
    model_group.add_argument('--seq_len', type=int, default=256)
    model_group.add_argument('--d_model', type=int, default=768)
    model_group.add_argument('--num_layers', type=int, default=12)
    model_group.add_argument('--num_heads', type=int, default=12)
    model_group.add_argument('--d_ff', type=int, default=3072)
    
    bench_group = parser.add_argument_group('Benchmarking Configuration')
    bench_group.add_argument('--batch_size', type=int, default=4)
    bench_group.add_argument('--warmup_steps', type=int, default=5)
    bench_group.add_argument('--iter_steps', type=int, default=10)
    bench_group.add_argument('--forward_only', action='store_true', default=False)
    
    device_group = parser.add_argument_group('Device Configuration')
    device_group.add_argument('--device', type=str, default='cuda')
    device_group.add_argument('--dtype', type=str, default='float32', choices=['float32', 'float16', 'bfloat16'])

    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()
    
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, switching to CPU.")
        args.device = 'cpu'
    
    device = torch.device(args.device)
    
    dtype_map = {
        'float32': torch.float32,
        'float16': torch.float16,
        'bfloat16': torch.bfloat16
    }
    dtype = dtype_map[args.dtype]
    
    d_ff = args.d_ff
    
    print("="*80)
    print("TRANSFORMER BENCHMARKING")
    print("="*80)

    print("\nBenchmark Configuration:")
    print(f"  Device:              {device}")
    print(f"  Data type:           {args.dtype}")
    print(f"  Warm-up steps:       {args.warmup_steps}")
    print(f"  Iteration steps:     {args.iter_steps}")
    print(f"  Mode:                {'Forward only' if args.forward_only else 'Forward + Backward'}")
    print()
    
    print("Initializing Transformer model...")
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.seq_len,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=d_ff,
        device=device,
        dtype=dtype
    )
    
    print(f"Moving model to {device} with dtype {args.dtype}...")
    model = model.to(device=device, dtype=dtype)
    
    model.train()
    
    # TESTING DEVICE PLACEMENT
    """
    print("Verifying device placement")
    param_devices = set(p.device for p in model.parameters())
    buffer_devices = set(b.device for b in model.buffers())
    all_devices = param_devices | buffer_devices
    
    if len(all_devices) > 1:
        print(f"ERROR: Model has tensors on multiple devices: {all_devices}")
        print("Checking each component:")
        for name, param in model.named_parameters():
            if param.device != device:
                print(f"  Parameter {name}: {param.device} (expected {device})")
        for name, buffer in model.named_buffers():
            if buffer.device != device:
                print(f"  Buffer {name}: {buffer.device} (expected {device})")
        raise RuntimeError("Model not fully on target device!")
    else:
        print(f"All model tensors on {device}")
    """
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    if dtype == torch.float32:
        dtype_size = 4
    elif dtype in [torch.float16, torch.bfloat16]:
        dtype_size = 2
    else:
        dtype_size = 4
    
    param_size_mb = total_params * dtype_size / (1024 ** 2)
    
    print(f"Model parameters: {total_params:,} (trainable: {trainable_params:,})")
    print(f"Model size: {param_size_mb:.2f} MB ({args.dtype})")
    print()
    
    print("Generating random batch")
    x, y = generate_random_batch(
        args.batch_size,
        args.seq_len,
        args.vocab_size,
        device
    )
    print(f"Input shape (x): {x.shape}")
    print(f"Target shape (y): {y.shape}")
    print()
    
    mean_time, std_time, times = run_benchmark(
        model,
        x,
        y,
        warmup_steps=args.warmup_steps,
        iter_steps=args.iter_steps,
        forward_only=args.forward_only,
        device=device
    )
    
    min_time = np.min(times)
    max_time = np.max(times)
    median_time = np.median(times)
    
    print()
    print("="*80)
    print("RESULTS")
    print("="*80)
    print("\nTiming Statistics:")
    print(f"  Mean:   {mean_time*1000:.2f} ms")
    print(f"  Std:    {std_time*1000:.2f} ms")
    print(f"  Min:    {min_time*1000:.2f} ms")
    print(f"  Max:    {max_time*1000:.2f} ms")
    print(f"  Median: {median_time*1000:.2f} ms")
    
    if device.type == 'cuda':
        print("\nGPU Memory:")
        print(f"  Allocated: {torch.cuda.memory_allocated(device) / 1e9:.2f} GB")
        print(f"  Reserved:  {torch.cuda.memory_reserved(device) / 1e9:.2f} GB")
        print(f"  Max allocated: {torch.cuda.max_memory_allocated(device) / 1e9:.2f} GB")
    
    print("\nComputational Estimates:")
    print(f"  Total parameters: {total_params:,}")
    
    # FLOP calculations
    # Per layer (forward pass):
    # - Attention: 2*B*T*d_model*d_model for QKV projections
    #              + 2*B*T^2*d_model for attention computation
    #              + 2*B*T*d_model*d_model for output projection
    # - FFN: 2*B*T*d_model*d_ff for each of two linear layers (simplified for SwiGLU)
    
    B = args.batch_size
    T = args.seq_len
    d = args.d_model
    
    # Attention FLOPs per layer
    qkv_proj_flops = 3 * 2 * B * T * d * d  # 3 projections (Q, K, V)
    attn_scores_flops = 2 * B * args.num_heads * T * T * (d // args.num_heads)
    attn_output_flops = 2 * B * args.num_heads * T * T * (d // args.num_heads)
    out_proj_flops = 2 * B * T * d * d
    attention_flops_per_layer = qkv_proj_flops + attn_scores_flops + attn_output_flops + out_proj_flops
    
    # FFN FLOPs per layer (simplified - actual SwiGLU has 3 matrices)
    ffn_flops_per_layer = 2 * 2 * B * T * d * d_ff
    
    total_flops_per_layer = attention_flops_per_layer + ffn_flops_per_layer
    total_flops = args.num_layers * total_flops_per_layer
    
    # Add embedding and final projection FLOPs
    final_proj_flops = 2 * B * T * d * args.vocab_size
    total_flops += final_proj_flops
    
    tflops = total_flops / 1e12
    tflops_per_sec = tflops / mean_time
    
    print(f"  Estimated FLOPs (forward): {tflops:.3f} TFLOPs")
    print(f"  Estimated throughput: {tflops_per_sec:.3f} TFLOPs/sec")
    print("\n" + "="*80)
    
    print("\nSUMMARY FOR ASSIGNMENT:")
    print(f"Configuration: L={args.num_layers}, d_model={args.d_model}, H={args.num_heads}, "
          f"B={args.batch_size}, T={args.seq_len}")
    print(f"Mode: {'Forward only' if args.forward_only else 'Forward + Backward'}")
    print(f"Warm-up steps: {args.warmup_steps}")
    print(f"Average time: {mean_time*1000:.2f} ms ± {std_time*1000:.2f} ms")
    print(f"Throughput: {args.batch_size * args.seq_len / mean_time:,.0f} tokens/sec")
    print("="*80)


if __name__ == '__main__':
    main()