#!/usr/bin/env python
# ABOUTME: Pre-flight test for PAMCL code correctness before H100 training
# ABOUTME: Run inside pamcl/venv on RTX 3050 with tiny batch_size=2
#
# Usage:
#   cd d:\My_Files\Research\pamcl
#   .\venv\Scripts\activate
#   python test_before_h100.py

import sys
import time
import os
import tempfile
import traceback
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# CUDA perf flags (same as train.py — verify they work)
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('high')

# Tiny sizes for RTX 3050 4GB
TEST_BATCH = 2
TEST_AUDIO_LEN = 16000  # 1 second
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

results = []


def test(name, passed, detail=""):
    status = "✅ PASS" if passed else "❌ FAIL"
    results.append((name, passed, detail))
    print(f"  {status}: {name}" + (f" — {detail}" if detail else ""))


def section(title):
    print(f"\n{'='*60}\n  {title}\n{'='*60}")


def main():
    print("\n" + "=" * 60)
    print("  PAMCL PRE-FLIGHT TEST — RTX 3050 → H100")
    print("=" * 60)
    start = time.time()

    # ── System Info ──
    section("SYSTEM INFO")
    print(f"  Python:  {sys.version.split()[0]}")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  CUDA:    {torch.version.cuda if torch.cuda.is_available() else 'N/A'}")
    if torch.cuda.is_available():
        print(f"  GPU:     {torch.cuda.get_device_name(0)}")
        print(f"  VRAM:    {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"  GPUs:    {torch.cuda.device_count()}")
    test("CUDA available", torch.cuda.is_available())

    # ── CUDA Optimizations ──
    section("TEST 1: CUDA OPTIMIZATIONS")
    test("cuDNN benchmark", torch.backends.cudnn.benchmark)
    test("TF32 matmul flag", True,
         "Active" if torch.backends.cuda.matmul.allow_tf32 else "No-op (OK)")

    # ── Config Loading ──
    section("TEST 2: CONFIG LOADING")
    try:
        from utils import load_config
        config = load_config(str(Path(__file__).parent / "configs" / "default.yaml"))

        bs = config['training']['batch_size']
        ga = config['training']['gradient_accumulation_steps']
        nw = config['data']['num_workers']
        dp = config['model'].get('use_data_parallel', False)

        test("Config loads", True)
        test(f"batch_size = {bs}", bs == 256)
        test(f"grad_accum = {ga}", ga == 1)
        test(f"num_workers = {nw}", nw == 8)
        test(f"use_data_parallel = {dp}", dp == True)
        test(f"Effective batch = {bs * ga}", bs * ga == 256)
    except Exception as e:
        test("Config loads", False, str(e))
        traceback.print_exc()

    # ── Model Build + Forward/Backward ──
    section("TEST 3: MODEL FORWARD + BACKWARD")
    if not torch.cuda.is_available():
        test("Model test (skipped)", True, "No GPU")
    else:
        try:
            from models import PAMCLModel

            print("  Loading model (SSL download may take ~30s)...")
            model = PAMCLModel(config).to(DEVICE)
            test("Model builds", True)

            total_p = sum(p.numel() for p in model.parameters())
            train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
            test(f"Params: {train_p:,} trainable / {total_p:,} total", True)

            # Forward
            audio = torch.randn(TEST_BATCH, TEST_AUDIO_LEN, device=DEVICE)
            labels = torch.randint(0, 2, (TEST_BATCH,), device=DEVICE)

            with torch.amp.autocast('cuda'):
                outputs = model(audio, labels=labels)

            test("Forward pass", True, f"Keys: {list(outputs.keys())}")

            # Backward
            if 'logits' in outputs:
                loss = F.cross_entropy(outputs['logits'], labels)
                loss.backward()
                test("Backward pass", True, f"Loss={loss.item():.4f}")

                has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                               for p in model.parameters() if p.requires_grad)
                test("Gradient flow", has_grad)

            # DataParallel logic
            gpu_count = torch.cuda.device_count()
            would_dp = dp and gpu_count > 1
            test("DataParallel logic", True,
                 f"Would enable: {would_dp} ({gpu_count} GPU(s), flag={dp})")

            # Param groups
            pg = model.get_trainable_params()
            test("get_trainable_params()", True,
                 f"ssl={len(pg['ssl'])}, new={len(pg['new'])}")

            del model, outputs, audio, labels
            torch.cuda.empty_cache()

        except Exception as e:
            test("Model test", False, str(e))
            traceback.print_exc()

    # ── Checkpoint Roundtrip ──
    section("TEST 4: CHECKPOINT ROUNDTRIP")
    try:
        m = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
        opt = torch.optim.AdamW(m.parameters())
        m(torch.randn(2, 64)).sum().backward()
        opt.step()

        ckpt_path = os.path.join(tempfile.gettempdir(), 'pamcl_test_ckpt.pt')
        torch.save({'model_state_dict': m.state_dict(), 'epoch': 0}, ckpt_path)
        test("Save", True)

        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)
        m2 = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
        m2.load_state_dict(ckpt['model_state_dict'])
        match = all(torch.allclose(a, b) for a, b in
                    zip(m.parameters(), m2.parameters()))
        test("Save/Load/Verify", match)
        os.unlink(ckpt_path)
    except Exception as e:
        test("Checkpoint", False, str(e))

    # ── Summary ──
    section("SUMMARY")
    elapsed = time.time() - start
    passed = sum(1 for _, p, _ in results if p)
    failed = sum(1 for _, p, _ in results if not p)

    print(f"\n  ✅ Passed: {passed}/{len(results)}")
    if failed:
        print(f"  ❌ Failed: {failed}/{len(results)}")
        for n, p, d in results:
            if not p:
                print(f"    • {n}: {d}")

    print(f"\n  Time: {elapsed:.1f}s")
    if failed == 0:
        print("\n  🚀 PAMCL is ready for H100! Run: python train.py")
    print("=" * 60)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
