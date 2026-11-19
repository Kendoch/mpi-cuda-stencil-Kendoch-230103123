import os
import sys
import json
import time
import hashlib
import platform
import threading
import subprocess
import re
import numpy as np
from numba import njit, types
from numba.extending import intrinsic
from llvmlite import ir

try:
    import resource
except ImportError:
    resource = None

GIT_USERNAME = "Kendoch"
NUM_THREADS = 4
ITERATIONS_PER_THREAD = 60_000_000

def detect_cache_line_size():
    path = "/sys/devices/system/cpu/cpu0/cache/index0/coherency_line_size"
    if os.path.exists(path):
        with open(path, "r") as f:
            return int(f.read().strip())
    if sys.platform == "darwin":
        try:
            return int(subprocess.check_output(
                ["sysctl", "-n", "hw.cachelinesize"],
                text=True
            ).strip())
        except Exception:
            pass
    return 64

def get_cpu_model():
    if os.path.exists("/proc/cpuinfo"):
        with open("/proc/cpuinfo", "r") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    return platform.processor() or platform.machine()

def init_system_context(git_user):
    assert git_user.strip()
    assert git_user != "REPLACE_WITH_YOUR_GIT_USERNAME"

    logical_cores = os.cpu_count() or 1
    available_cores = sorted(os.sched_getaffinity(0)) if hasattr(
        os, "sched_getaffinity"
    ) else list(range(logical_cores))

    if len(available_cores) < NUM_THREADS:
        raise RuntimeError(
            f"Need 4 available CPU cores, found {len(available_cores)}"
        )

    line_size = detect_cache_line_size()
    cpu_model = get_cpu_model()
    cpu_sig = f"{platform.machine()}::{cpu_model}::{logical_cores}"
    digest = hashlib.sha256(
        f"{git_user.strip()}::{cpu_sig}".encode()
    ).hexdigest()

    instance_seed = int(digest[:16], 16)

    context = {
        "git_username": git_user,
        "cpu_model": cpu_model,
        "cpu_signature": cpu_sig,
        "logical_cores": logical_cores,
        "available_cores": available_cores,
        "cache_line_size": line_size,
        "instance_seed": instance_seed,
        "num_threads": NUM_THREADS,
        "iterations_per_thread": ITERATIONS_PER_THREAD
    }

    return context

@intrinsic
def compiler_memory_barrier(typingctx):
    sig = types.void()

    def codegen(context, builder, signature, args):
        asm = ir.InlineAsm(
            ir.FunctionType(ir.VoidType(), []),
            "",
            "~{memory}",
            side_effect=True
        )
        builder.call(asm, [])
        return context.get_dummy_value()

    return sig, codegen

@njit(nogil=True)
def worker_strided_accumulate(buffer, slot, iters, thread_seed):
    delta = np.uint64(thread_seed % 7 + 1)

    for i in range(iters):
        buffer[slot] = buffer[slot] + delta
        compiler_memory_barrier()

@njit(nogil=True)
def worker_private_accumulate(buffer, slot, iters, thread_seed):
    delta = np.uint64(thread_seed % 7 + 1)
    accumulator = np.uint64(0)

    for i in range(iters):
        accumulator += delta

    buffer[slot] = accumulator

def aligned_buffer(length):
    raw = np.zeros(length + 16, dtype=np.uint64)
    offset = (-raw.ctypes.data) % 64
    start = offset // 8
    result = raw[start:start + length]

    assert result.ctypes.data % 64 == 0

    return result

def get_context_switches():
    if resource is None:
        return None

    usage = resource.getrusage(resource.RUSAGE_SELF)

    return {
        "voluntary": usage.ru_nvcsw,
        "involuntary": usage.ru_nivcsw
    }

def run_stride_benchmark(context, stride, private=False):
    length = NUM_THREADS * stride + 8
    buffer = aligned_buffer(length)
    cores = context["available_cores"][:NUM_THREADS]
    errors = []

    def thread_worker(tid):
        try:
            if hasattr(os, "sched_setaffinity") and hasattr(
                threading, "get_native_id"
            ):
                os.sched_setaffinity(
                    threading.get_native_id(),
                    {cores[tid]}
                )

            slot = tid * stride
            seed = context["instance_seed"] % 1_000_003 + tid

            if private:
                worker_private_accumulate(
                    buffer, slot, ITERATIONS_PER_THREAD, seed
                )
            else:
                worker_strided_accumulate(
                    buffer, slot, ITERATIONS_PER_THREAD, seed
                )

        except Exception as e:
            errors.append(str(e))

    before = get_context_switches()
    start = time.perf_counter_ns()

    threads = [
        threading.Thread(target=thread_worker, args=(tid,))
        for tid in range(NUM_THREADS)
    ]

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    end = time.perf_counter_ns()
    after = get_context_switches()

    if errors:
        raise RuntimeError(str(errors))

    values = [
        int(buffer[tid * stride])
        for tid in range(NUM_THREADS)
    ]

    expected = [
        ITERATIONS_PER_THREAD *
        ((context["instance_seed"] % 1_000_003 + tid) % 7 + 1)
        for tid in range(NUM_THREADS)
    ]

    if values != expected:
        raise AssertionError(
            f"State mismatch: {values} != {expected}"
        )

    elapsed_ns = end - start
    total_updates = NUM_THREADS * ITERATIONS_PER_THREAD

    switches = None

    if before is not None and after is not None:
        switches = {
            "voluntary": after["voluntary"] - before["voluntary"],
            "involuntary": after["involuntary"] - before["involuntary"]
        }

    return {
        "layout": "private" if private else f"stride_{stride}",
        "stride": stride,
        "elapsed_ns": elapsed_ns,
        "elapsed_ms": elapsed_ns / 1_000_000,
        "ns_per_update": elapsed_ns / total_updates,
        "total_updates": total_updates,
        "values": values,
        "expected_values": expected,
        "context_switches": switches,
        "aligned_64_bytes": buffer.ctypes.data % 64 == 0,
        "parity_passed": True
    }

def audit_llvm_and_parity(results):
    signature = worker_strided_accumulate.signatures[0]
    llvm_ir = worker_strided_accumulate.inspect_llvm(signature)

    store_matches = re.findall(
        r"\bstore\s+i64\b",
        llvm_ir
    )

    load_matches = re.findall(
        r"\bload\s+i64\b",
        llvm_ir
    )

    barrier_count = llvm_ir.count('asm sideeffect')

    loop_store_detected = (
        len(store_matches) > 0 and barrier_count > 0
    )

    reference = results["stride_1"]["values"]

    parity = all(
        result["values"] == reference
        for result in results.values()
    )

    audit = {
        "llvm_native_store_count": len(store_matches),
        "llvm_native_load_count": len(load_matches),
        "llvm_memory_barrier_count": barrier_count,
        "llvm_loop_store_detected": loop_store_detected,
        "native_compilation": True,
        "exact_state_parity": parity,
        "verification_passed": bool(
            parity and loop_store_detected
        )
    }

    return audit, llvm_ir

def main():
    context = init_system_context(GIT_USERNAME)

    print("SYSTEM CONTEXT")
    print(json.dumps(context, indent=2))

    warmup = aligned_buffer(64)
    worker_strided_accumulate(warmup, 0, 10, 123)
    worker_private_accumulate(warmup, 8, 10, 123)

    results = {}

    for stride in [1, 4, 8, 16]:
        print(f"\nRunning stride = {stride}", flush=True)

        result = run_stride_benchmark(
            context,
            stride,
            private=False
        )

        results[f"stride_{stride}"] = result

        print(
            f"stride={stride} "
            f"elapsed_ns={result['elapsed_ns']} "
            f"ns_per_update={result['ns_per_update']:.6f}"
        )

    print("\nRunning register privatization", flush=True)

    results["private"] = run_stride_benchmark(
        context,
        8,
        private=True
    )

    slowdown = (
        results["stride_1"]["elapsed_ns"] /
        results["stride_8"]["elapsed_ns"]
    )

    private_speedup = (
        results["stride_1"]["elapsed_ns"] /
        results["private"]["elapsed_ns"]
    )

    audit, llvm_ir = audit_llvm_and_parity(results)

    manifest = {
        **context,
        "benchmark_results": results,
        "slowdown_stride1_vs_stride8": slowdown,
        "speedup_private_vs_stride1": private_speedup,
        "llvm_audit": audit,
        "verification_passed": audit["verification_passed"]
    }

    with open("cache_lab_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    with open("worker_llvm_ir.txt", "w") as f:
        f.write(llvm_ir)

    print("\nBENCHMARK RESULTS")

    for name, result in results.items():
        print(
            f"{name}: "
            f"{result['elapsed_ns']} ns, "
            f"{result['ns_per_update']:.6f} ns/update"
        )

    print(f"\nSlowdown stride 1 vs 8: {slowdown:.4f}x")
    print(f"Private speedup vs stride 1: {private_speedup:.4f}x")

    print("\nLLVM AUDIT")
    print(json.dumps(audit, indent=2))

    print("\nVERIFICATION")
    print("PASS" if audit["verification_passed"] else "FAIL")

    print("\nManifest saved: cache_lab_manifest.json")

if __name__ == "__main__":
    main()