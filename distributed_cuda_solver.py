import os
import sys
import json
import math
import time
import re
import hashlib
import subprocess
import numpy as np
from mpi4py import MPI
from numba import cuda, float64

TILE_DIM = 16
GLOBAL_NX = 512
GLOBAL_NY = 512
STEPS = 400
LAMBDA_DAMP = 0.15
DIRECTIONS = (0, 1, 2, 3)
TAGS = (100, 101, 102, 103)
OPPOSITE = (1, 0, 3, 2)


def get_gpu_uuid():
    output = subprocess.check_output(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], text=True)
    return output.strip().splitlines()[0].strip()


def derive_student_seed(github_user, gpu_uuid):
    digest = hashlib.sha256(f"{github_user.strip()}::{gpu_uuid}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def init_distributed_context(github_user):
    world = MPI.COMM_WORLD
    rank = world.Get_rank()
    size = world.Get_size()
    if size != 4:
        raise RuntimeError(f"Expected 4 MPI ranks, got {size}")
    if not github_user.strip():
        raise RuntimeError("Specify your GitHub username")
    cuda.select_device(0)
    gpu_uuid = world.bcast(get_gpu_uuid() if rank == 0 else None, root=0)
    instance_seed = derive_student_seed(github_user, gpu_uuid)
    cart = world.Create_cart([2, 2], periods=[False, False], reorder=False)
    up, down = cart.Shift(0, 1)
    left, right = cart.Shift(1, 1)
    return dict(world=world, cart=cart, rank=rank, size=size, coords=cart.Get_coords(rank),
                neighbors=(up, down, left, right), gpu_uuid=gpu_uuid, instance_seed=instance_seed)


def initialize_subdomains(ctx):
    ny = GLOBAL_NY // 2
    nx = GLOBAL_NX // 2
    dy = 1.0 / (GLOBAL_NY + 1)
    dx = 1.0 / (GLOBAL_NX + 1)
    row_rank, col_rank = ctx["coords"]
    y = ((row_rank * ny + np.arange(ny + 2, dtype=np.float64)) * dy)[:, None]
    x = ((col_rank * nx + np.arange(nx + 2, dtype=np.float64)) * dx)[None, :]
    seed = ctx["instance_seed"]
    phase = (seed % 1000003) / 1000003.0
    x0 = 0.30 + 0.10 * phase
    y0 = 0.33 + 0.08 * phase
    x1 = 0.68 - 0.08 * phase
    y1 = 0.65 - 0.10 * phase
    kappa = 0.08 + 0.16 * (0.5 + 0.5 * np.sin(2.0 * np.pi * x) * np.cos(2.0 * np.pi * y))
    pulse0 = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * 0.07 ** 2))
    pulse1 = np.exp(-((x - x1) ** 2 + (y - y1) ** 2) / (2 * 0.055 ** 2))
    initial = (0.7 + 0.2 * phase) * pulse0 + (0.4 + 0.1 * phase) * pulse1
    source = 0.01 * (np.exp(-((x - 0.46) ** 2 + (y - 0.48) ** 2) / (2 * 0.045 ** 2))
                     - np.exp(-((x - 0.54) ** 2 + (y - 0.52) ** 2) / (2 * 0.045 ** 2)))
    initial = np.ascontiguousarray(initial, dtype=np.float64)
    kappa = np.ascontiguousarray(kappa, dtype=np.float64)
    source = np.ascontiguousarray(source, dtype=np.float64)
    if ctx["neighbors"][0] == MPI.PROC_NULL:
        initial[0, :] = 0.0
    if ctx["neighbors"][1] == MPI.PROC_NULL:
        initial[-1, :] = 0.0
    if ctx["neighbors"][2] == MPI.PROC_NULL:
        initial[:, 0] = 0.0
    if ctx["neighbors"][3] == MPI.PROC_NULL:
        initial[:, -1] = 0.0
    return initial, kappa, source, dx, dy


@cuda.jit
def pack_halos_kernel(u, packed, direction, ny, nx):
    t = cuda.grid(1)
    n = nx if direction < 2 else ny
    if t < n:
        if direction == 0:
            packed[t] = u[1, t + 1]
        elif direction == 1:
            packed[t] = u[ny, t + 1]
        elif direction == 2:
            packed[t] = u[t + 1, 1]
        else:
            packed[t] = u[t + 1, nx]


@cuda.jit
def unpack_halos_kernel(u, packed, direction, ny, nx):
    t = cuda.grid(1)
    n = nx if direction < 2 else ny
    if t < n:
        if direction == 0:
            u[0, t + 1] = packed[t]
        elif direction == 1:
            u[ny + 1, t + 1] = packed[t]
        elif direction == 2:
            u[t + 1, 0] = packed[t]
        else:
            u[t + 1, nx + 1] = packed[t]


@cuda.jit
def tiled_stencil_kernel(u, v, kappa, source, dt, inv_dx2, inv_dy2,
                         r_start, r_end, c_start, c_end):
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    i = r_start + cuda.blockIdx.y * TILE_DIM + ty
    j = c_start + cuda.blockIdx.x * TILE_DIM + tx
    sy = ty + 1
    sx = tx + 1
    smem = cuda.shared.array((18, 18), dtype=float64)
    if i < u.shape[0] and j < u.shape[1]:
        smem[sy, sx] = u[i, j]
    else:
        smem[sy, sx] = 0.0
    if ty == 0 and i > 0 and j < u.shape[1]:
        smem[0, sx] = u[i - 1, j]
    if ty == 15 and i + 1 < u.shape[0] and j < u.shape[1]:
        smem[17, sx] = u[i + 1, j]
    if tx == 0 and j > 0 and i < u.shape[0]:
        smem[sy, 0] = u[i, j - 1]
    if tx == 15 and j + 1 < u.shape[1] and i < u.shape[0]:
        smem[sy, 17] = u[i, j + 1]
    cuda.syncthreads()
    if i <= r_end and j <= c_end:
        center = smem[sy, sx]
        kp = 0.5 * (kappa[i, j] + kappa[i + 1, j])
        km = 0.5 * (kappa[i, j] + kappa[i - 1, j])
        kr = 0.5 * (kappa[i, j] + kappa[i, j + 1])
        kl = 0.5 * (kappa[i, j] + kappa[i, j - 1])
        diffusion_y = (kp * (smem[sy + 1, sx] - center) - km * (center - smem[sy - 1, sx])) * inv_dy2
        diffusion_x = (kr * (smem[sy, sx + 1] - center) - kl * (center - smem[sy, sx - 1])) * inv_dx2
        v[i, j] = center + dt * (diffusion_y + diffusion_x - LAMBDA_DAMP * center ** 3 + source[i, j])


def allocate_halos(ny, nx):
    result = []
    for d in DIRECTIONS:
        n = nx if d < 2 else ny
        result.append(dict(dsend=cuda.device_array(n, dtype=np.float64),
                           drecv=cuda.device_array(n, dtype=np.float64),
                           hsend=cuda.pinned_array(n, dtype=np.float64),
                           hrecv=cuda.pinned_array(n, dtype=np.float64)))
    return result


def launch_box(u, v, kappa, source, dt, dx, dy, box, stream):
    rs, re, cs, ce = box
    grid = (math.ceil((ce - cs + 1) / 16), math.ceil((re - rs + 1) / 16))
    tiled_stencil_kernel[grid, (16, 16), stream](u, v, kappa, source, dt, 1.0 / dx ** 2,
                                                   1.0 / dy ** 2, rs, re, cs, ce)


def exchange_halos_async(ctx, u, buffers, stream_boundary):
    ny, nx = u.shape[0] - 2, u.shape[1] - 2
    active = [d for d in DIRECTIONS if ctx["neighbors"][d] != MPI.PROC_NULL]
    for d in active:
        b = buffers[d]
        pack_halos_kernel[math.ceil(len(b["hsend"]) / 128), 128, stream_boundary](u, b["dsend"], d, ny, nx)
        b["dsend"].copy_to_host(b["hsend"], stream=stream_boundary)
    stream_boundary.synchronize()
    requests = []
    for d in active:
        requests.append(ctx["cart"].Irecv(buffers[d]["hrecv"], source=ctx["neighbors"][d], tag=TAGS[OPPOSITE[d]]))
    for d in active:
        requests.append(ctx["cart"].Isend(buffers[d]["hsend"], dest=ctx["neighbors"][d], tag=TAGS[d]))
    return requests, active


def complete_halos(u, buffers, requests, active, stream_boundary):
    MPI.Request.Waitall(requests)
    ny, nx = u.shape[0] - 2, u.shape[1] - 2
    for d in active:
        b = buffers[d]
        b["drecv"].copy_to_device(b["hrecv"], stream=stream_boundary)
        unpack_halos_kernel[math.ceil(len(b["hrecv"]) / 128), 128, stream_boundary](u, b["drecv"], d, ny, nx)


def global_diagnostics(ctx, u, previous, dx, dy):
    local = u.copy_to_host()[1:-1, 1:-1]
    local_old = previous.copy_to_host()[1:-1, 1:-1]
    local_energy = float(np.sum(local) * dx * dy)
    local_residual = float(np.sum((local - local_old) ** 2) * dx * dy)
    energy = ctx["cart"].allreduce(local_energy, op=MPI.SUM)
    residual = math.sqrt(ctx["cart"].allreduce(local_residual, op=MPI.SUM))
    return energy, residual


def run_mode(ctx, initial, kappa_host, source_host, dx, dy, dt, overlapped):
    ny, nx = initial.shape[0] - 2, initial.shape[1] - 2
    u = cuda.to_device(initial)
    v = cuda.to_device(np.zeros_like(initial))
    kappa = cuda.to_device(kappa_host)
    source = cuda.to_device(source_host)
    buffers = allocate_halos(ny, nx)
    stream_boundary = cuda.stream()
    stream_interior = cuda.stream()
    trace = []
    e0, r0 = global_diagnostics(ctx, u, u, dx, dy)
    trace.append(dict(step=0, energy=e0, residual=r0))
    cart = ctx["cart"]
    cuda.synchronize()
    cart.Barrier()
    start = MPI.Wtime()
    gpu_start = cuda.event(timing=True)
    gpu_stop = cuda.event(timing=True)
    gpu_start.record(stream_boundary)
    for step in range(1, STEPS + 1):
        requests, active = exchange_halos_async(ctx, u, buffers, stream_boundary)
        if overlapped:
            launch_box(u, v, kappa, source, dt, dx, dy, (2, ny - 1, 2, nx - 1), stream_interior)
        complete_halos(u, buffers, requests, active, stream_boundary)
        if not overlapped:
            launch_box(u, v, kappa, source, dt, dx, dy, (2, ny - 1, 2, nx - 1), stream_boundary)
        launch_box(u, v, kappa, source, dt, dx, dy, (1, 1, 1, nx), stream_boundary)
        launch_box(u, v, kappa, source, dt, dx, dy, (ny, ny, 1, nx), stream_boundary)
        launch_box(u, v, kappa, source, dt, dx, dy, (2, ny - 1, 1, 1), stream_boundary)
        launch_box(u, v, kappa, source, dt, dx, dy, (2, ny - 1, nx, nx), stream_boundary)
        stream_boundary.synchronize()
        stream_interior.synchronize()
        if step % 100 == 0:
            energy, residual = global_diagnostics(ctx, v, u, dx, dy)
            trace.append(dict(step=step, energy=energy, residual=residual))
        u, v = v, u
    gpu_stop.record(stream_boundary)
    gpu_stop.synchronize()
    cuda.synchronize()
    elapsed = (MPI.Wtime() - start) * 1000.0
    wall_ms = cart.allreduce(elapsed, op=MPI.MAX)
    cuda_event_ms = cart.allreduce(cuda.event_elapsed_time(gpu_start, gpu_stop), op=MPI.MAX)
    return u.copy_to_host(), trace, wall_ms, cuda_event_ms, tiled_stencil_kernel


def audit_ptx(kernel):
    ptx = kernel.inspect_asm(kernel.signatures[0])
    counts = dict(st_shared_f64=len(re.findall(r"\bst\.shared\.f64\b", ptx)),
                  ld_shared_f64=len(re.findall(r"\bld\.shared\.f64\b", ptx)),
                  bar_sync=len(re.findall(r"\bbar\.sync\b", ptx)))
    return counts, all(counts.values())


def run_solver(github_user):
    ctx = init_distributed_context(github_user)
    initial, kappa, source, dx, dy = initialize_subdomains(ctx)
    local_max = float(np.max(kappa))
    kappa_max = ctx["cart"].allreduce(local_max, op=MPI.MAX)
    dt_max = min(dx, dy) ** 2 / (4.0 * kappa_max)
    dt = 0.8 * dt_max
    if ctx["rank"] == 0:
        print(f"github_username={github_user} mpi_ranks=4 cart_dims=2x2 gpu_uuid={ctx['gpu_uuid']}", flush=True)
        print(f"instance_seed={ctx['instance_seed']} kappa_max={kappa_max:.16e} dx={dx:.16e} dt_max={dt_max:.16e} dt={dt:.16e}", flush=True)
    serialized, trace_s, ms_s, event_s, kernel = run_mode(ctx, initial, kappa, source, dx, dy, dt, False)
    overlapped, trace_o, ms_o, event_o, kernel = run_mode(ctx, initial, kappa, source, dx, dy, dt, True)
    local_diff = float(np.max(np.abs(serialized[1:-1, 1:-1] - overlapped[1:-1, 1:-1])))
    parity = ctx["cart"].allreduce(local_diff, op=MPI.MAX)
    counts, passed = audit_ptx(kernel)
    global_passed = bool(ctx["cart"].allreduce(int(passed), op=MPI.MIN))
    if ctx["rank"] == 0:
        for item in trace_o:
            print(f"step={item['step']} energy={item['energy']:.16e} residual={item['residual']:.16e}", flush=True)
        print(f"serialized_ms={ms_s:.4f} overlapped_ms={ms_o:.4f} speedup={ms_s / ms_o:.6f}", flush=True)
        print(f"serialized_cuda_event_ms={event_s:.4f} overlapped_cuda_event_ms={event_o:.4f}", flush=True)
        print(f"parity_max_diff={parity:.16e} ptx_counts={counts} ptx_audit_passed={global_passed}", flush=True)
        manifest = dict(github_username=github_user, gpu_uuid=ctx["gpu_uuid"],
                        instance_seed=ctx["instance_seed"], mpi_ranks=4, cart_dims=[2, 2],
                        global_grid=[GLOBAL_NY, GLOBAL_NX], local_padded_grid=[258, 258],
                        steps=STEPS, dx=dx, dy=dy, kappa_max=kappa_max, dt_max=dt_max, dt=dt,
                        serialized_ms=ms_s, overlapped_ms=ms_o,
                        serialized_cuda_event_ms=event_s, overlapped_cuda_event_ms=event_o,
                        speedup=ms_s / ms_o, parity_max_diff=parity,
                        ptx_counts=counts, ptx_audit_passed=global_passed,
                        energy_trace=trace_o, serialized_energy_trace=trace_s)
        with open("submission_manifest.json", "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)
        if not global_passed or parity >= 1e-12:
            raise RuntimeError("PTX or parity verification failed")
        print("PASS: manifest written", flush=True)
    ctx["cart"].Barrier()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python distributed_cuda_solver.py YOUR_GITHUB_USERNAME")
    run_solver(sys.argv[1])
