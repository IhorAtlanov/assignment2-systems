#!/usr/bin/env python3
"""
Benchmark script with NVTX annotations for Nsight Systems profiling.

Example usage:
    uv run nsys profile -o result --capture-range=cudaProfilerApi python benchmark_nvtx.py \
        --num_layers 12 --d_model 768 --num_heads 12 --d_ff 3072 --batch_size 4 --seq_len 512
"""

import argparse
import torch
import torch.cuda.nvtx as nvtx
import numpy as np
from timeit import default_timer as timer

from cs336_basics.TransformerLM.transformer_lm import TransformerLM
from cs336_basics.Cross_entropy_loss_AdamW.cross_entropy import run_cross_entropy


def generate_random_batch(batch_size: int, seq_len: int, vocab_size: int = 10000, device: str = 'cuda') -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    y = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    return x, y


def benchmark_step(model: TransformerLM, x: torch.Tensor, y: torch.Tensor, forward_only: bool = False) -> None:
    with nvtx.range("forward_pass"):
        forward = model(x)
    
    if not forward_only:
        with nvtx.range("loss_computation"):
            loss = run_cross_entropy(forward, y)
        
        with nvtx.range("backward_pass"):
            loss.backward()
        
        with nvtx.range("zero_grad"):
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
    
    # Warm-up phase (wrapped in NVTX range for easy filtering)
    with nvtx.range("warmup_phase"):
        print(f"Running {warmup_steps} warm-up steps...")
        for i in range(warmup_steps):
            with nvtx.range(f"warmup_step_{i}"):
                benchmark_step(model, x, y, forward_only)
                if device.type == 'cuda':
                    torch.cuda.synchronize(device)
            print(f"  Warm-up {i+1}/{warmup_steps} complete")
    
    # Start CUDA profiler after warm-up (for --capture-range=cudaProfilerApi)
    torch.cuda.cudart().cudaProfilerStart()
    
    print(f"\nRunning {iter_steps} iteration steps...")
    
    # Iteration phase
    times = []
    with nvtx.range("benchmark_phase"):
        for i in range(iter_steps):
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            
            start = timer()
            
            with nvtx.range(f"iteration_step_{i}"):
                benchmark_step(model, x, y, forward_only)
            
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            
            end = timer()
            
            elapsed = end - start
            times.append(elapsed)
            print(f"  Step {i+1}/{iter_steps}: {elapsed*1000:.2f} ms")
    
    # Stop CUDA profiler
    torch.cuda.cudart().cudaProfilerStop()
    
    mean_time = np.mean(times)
    std_time = np.std(times)
    
    return mean_time, std_time, times


def get_parser():
    parser = argparse.ArgumentParser(
        description='Benchmark Transformer model with NVTX annotations',
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
    print("TRANSFORMER BENCHMARKING (WITH NVTX ANNOTATIONS)")
    print("="*80)

    print("\nBenchmark Configuration:")
    print(f"  Device:              {device}")
    print(f"  Data type:           {args.dtype}")
    print(f"  Warm-up steps:       {args.warmup_steps}")
    print(f"  Iteration steps:     {args.iter_steps}")
    print(f"  Mode:                {'Forward only' if args.forward_only else 'Forward + Backward'}")
    print()
    
    with nvtx.range("model_initialization"):
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
        model = model.to(device=device, dtype=dtype)
        model.train()
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    print()
    
    with nvtx.range("data_generation"):
        print("Generating random batch")
        x, y = generate_random_batch(args.batch_size, args.seq_len, args.vocab_size, device)
        print(f"Input shape (x): {x.shape}")
        print(f"Target shape (y): {y.shape}")
    print()
    
    mean_time, std_time = run_benchmark(
        model, x, y,
        warmup_steps=args.warmup_steps,
        iter_steps=args.iter_steps,
        forward_only=args.forward_only,
        device=device
    )
    
    print()
    print("="*80)
    print("RESULTS")
    print("="*80)
    print(f"\nMean: {mean_time*1000:.2f} ms ± {std_time*1000:.2f} ms")
    print(f"Throughput: {args.batch_size * args.seq_len / mean_time:,.0f} tokens/sec")
    print("="*80)


if __name__ == '__main__':
    main()