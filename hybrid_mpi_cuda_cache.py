import os
import sys
import re
import json
import time
import math
import hashlib
import platform
import subprocess
import numpy as np
from mpi4py import MPI
from numba import cuda, njit, float64, types
from numba.extending import intrinsic
from llvmlite import ir

TILE_DIM = 16
GLOBAL_NX = 512
GLOBAL_NY = 512
LOCAL_N = 256
STENCIL_STEPS = 300
SHM_ITERS = 40_000_000
LAMBDA_DAMP = 0.15
DIRECTIONS = ('up', 'down', 'left', 'right')
OPPOSITE = {'up': 'down', 'down': 'up', 'left': 'right', 'right': 'left'}
TAGS = {'up': 20, 'down': 21, 'left': 22, 'right': 23}

@cuda.jit
def pack_halos_kernel(u, output, direction):
    t = cuda.grid(1)
    if t < LOCAL_N:
        if direction == 0:
            output[t] = u[1, t + 1]
        elif direction == 1:
            output[t] = u[LOCAL_N, t + 1]
        elif direction == 2:
            output[t] = u[t + 1, 1]
        else:
            output[t] = u[t + 1, LOCAL_N]

@cuda.jit
def unpack_halos_kernel(u, incoming, direction):
    t = cuda.grid(1)
    if t < LOCAL_N:
        if direction == 0:
            u[0, t + 1] = incoming[t]
        elif direction == 1:
            u[LOCAL_N + 1, t + 1] = incoming[t]
        elif direction == 2:
            u[t + 1, 0] = incoming[t]
        else:
            u[t + 1, LOCAL_N + 1] = incoming[t]

@cuda.jit
def tiled_stencil_kernel(u, kappa, source, result, dt, inv_dx2, region):
    ty = cuda.threadIdx.y
    tx = cuda.threadIdx.x
    by = cuda.blockIdx.y * TILE_DIM
    bx = cuda.blockIdx.x * TILE_DIM
    row = by + ty + 1
    col = bx + tx + 1
    tile = cuda.shared.array((18, 18), dtype=float64)
    tile[ty + 1, tx + 1] = u[row, col]
    if ty == 0:
        tile[0, tx + 1] = u[row - 1, col]
    if ty == TILE_DIM - 1:
        tile[17, tx + 1] = u[row + 1, col]
    if tx == 0:
        tile[ty + 1, 0] = u[row, col - 1]
    if tx == TILE_DIM - 1:
        tile[ty + 1, 17] = u[row, col + 1]
    cuda.syncthreads()
    if row <= LOCAL_N and col <= LOCAL_N:
        deep = row >= 2 and row <= LOCAL_N - 1 and col >= 2 and col <= LOCAL_N - 1
        if (region == 0 and deep) or (region == 1 and not deep) or region == 2:
            center = tile[ty + 1, tx + 1]
            laplace = (tile[ty, tx + 1] + tile[ty + 2, tx + 1] + tile[ty + 1, tx] + tile[ty + 1, tx + 2] - 4.0 * center) * inv_dx2
            result[row, col] = center + dt * (kappa[row, col] * laplace + source[row, col] - LAMBDA_DAMP * center)

@intrinsic
def memory_barrier(typingctx):
    signature = types.void()
    def codegen(context, builder, signature, args):
        instruction = ir.InlineAsm(ir.FunctionType(ir.VoidType(), []), '', '~{memory}', side_effect=True)
        builder.call(instruction, [])
        return context.get_dummy_value()
    return signature, codegen

@njit(nogil=True)
def host_shm_accumulate(shm_view, slot, iterations):
    for i in range(iterations):
        shm_view[slot] = shm_view[slot] + np.uint64(1)
        memory_barrier()


def gpu_uuid():
    return subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip().splitlines()[0]


def cpu_line_size():
    p = '/sys/devices/system/cpu/cpu0/cache/index0/coherency_line_size'
    if os.path.exists(p):
        with open(p, encoding='utf-8') as f:
            return int(f.read().strip())
    return 64


def init_hybrid_context(github_user):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    if comm.Get_size() != 4:
        raise RuntimeError('Exactly four MPI processes are required')
    cuda.select_device(0)
    cart = comm.Create_cart([2, 2], periods=[False, False], reorder=False)
    up, down = cart.Shift(0, 1)
    left, right = cart.Shift(1, 1)
    neighbors = dict(zip(DIRECTIONS, (up, down, left, right)))
    node = comm.Split_type(MPI.COMM_TYPE_SHARED)
    if node.Get_size() != 4:
        raise RuntimeError('All four MPI ranks must reside on one shared-memory node')
    node_rank = node.Get_rank()
    win = MPI.Win.Allocate_shared(64 * 8 + 64 if node_rank == 0 else 0, 1, comm=node)
    buffer, _ = win.Shared_query(0)
    full = np.ndarray(buffer=buffer, dtype=np.uint8, shape=(64 * 8 + 64,))
    offset = (-full.ctypes.data) % 64
    shm = np.ndarray(shape=(64,), buffer=buffer, dtype=np.uint64, offset=offset)
    if shm.ctypes.data % 64:
        raise RuntimeError('The shared buffer is not 64-byte aligned')
    if node_rank == 0:
        shm.fill(0)
    node.Barrier()
    uuid = comm.bcast(gpu_uuid() if rank == 0 else None, root=0)
    cpu_sig = platform.machine() + '::' + platform.processor() + '::' + str(os.cpu_count())
    signature = f'{github_user}::{cpu_sig}::{uuid}'
    seed = int.from_bytes(hashlib.sha256(signature.encode()).digest()[:8], 'big')
    return dict(comm=comm, cart=cart, node=node, win=win, shm=shm, rank=rank, node_rank=node_rank, neighbors=neighbors, gpu_uuid=uuid, cpu_signature=cpu_sig, instance_seed=seed, cache_line_size=cpu_line_size(), shared_address=int(shm.ctypes.data))


def initialize_fields(ctx):
    y_rank, x_rank = ctx['cart'].Get_coords(ctx['rank'])
    yy = np.arange(y_rank * LOCAL_N + 1, (y_rank + 1) * LOCAL_N + 1, dtype=np.float64)[:, None]
    xx = np.arange(x_rank * LOCAL_N + 1, (x_rank + 1) * LOCAL_N + 1, dtype=np.float64)[None, :]
    x = xx / 513.0
    y = yy / 513.0
    u = np.zeros((258, 258), dtype=np.float64)
    kappa = np.zeros_like(u)
    source = np.zeros_like(u)
    u[1:257, 1:257] = np.sin(np.pi * x) * np.sin(np.pi * y)
    kappa[1:257, 1:257] = 0.7 + 0.2 * np.sin(2.0 * np.pi * x) ** 2 + 0.1 * np.cos(2.0 * np.pi * y) ** 2
    local_max = float(kappa.max())
    kmax = ctx['cart'].allreduce(local_max, op=MPI.MAX)
    dx = 1.0 / 513.0
    dtmax = dx * dx / (4.0 * kmax)
    dt = 0.8 * dtmax
    return dict(u=cuda.to_device(u), kappa=cuda.to_device(kappa), source=cuda.to_device(source), dt=dt, dt_max=dtmax, dx=dx, kappa_max=kmax)


def energy(ctx, device_array):
    local = device_array.copy_to_host()[1:257, 1:257]
    return ctx['cart'].allreduce(float(np.sum(local * local, dtype=np.float64)), op=MPI.SUM)


def make_buffers():
    send_host = {d: cuda.pinned_array(LOCAL_N, dtype=np.float64) for d in DIRECTIONS}
    recv_host = {d: cuda.pinned_array(LOCAL_N, dtype=np.float64) for d in DIRECTIONS}
    send_dev = {d: cuda.device_array(LOCAL_N, dtype=np.float64) for d in DIRECTIONS}
    recv_dev = {d: cuda.device_array(LOCAL_N, dtype=np.float64) for d in DIRECTIONS}
    return send_host, recv_host, send_dev, recv_dev


def step(ctx, u, dest, kappa, source, dt, dx, buffers, stream_interior, stream_boundary, overlapped):
    send_host, recv_host, send_dev, recv_dev = buffers
    neighbors = ctx['neighbors']
    cart = ctx['cart']
    requests = []
    for d in DIRECTIONS:
        if neighbors[d] != MPI.PROC_NULL:
            requests.append(cart.Irecv(recv_host[d], source=neighbors[d], tag=TAGS[OPPOSITE[d]]))
    for j, d in enumerate(DIRECTIONS):
        if neighbors[d] != MPI.PROC_NULL:
            pack_halos_kernel[2, 128, stream_boundary](u, send_dev[d], j)
            send_dev[d].copy_to_host(send_host[d], stream=stream_boundary)
    if overlapped:
        tiled_stencil_kernel[(16, 16), (16, 16), stream_interior](u, kappa, source, dest, dt, 1.0 / (dx * dx), 0)
    stream_boundary.synchronize()
    for d in DIRECTIONS:
        if neighbors[d] != MPI.PROC_NULL:
            requests.append(cart.Isend(send_host[d], dest=neighbors[d], tag=TAGS[d]))
    MPI.Request.Waitall(requests)
    for j, d in enumerate(DIRECTIONS):
        if neighbors[d] != MPI.PROC_NULL:
            recv_dev[d].copy_to_device(recv_host[d], stream=stream_boundary)
            unpack_halos_kernel[2, 128, stream_boundary](u, recv_dev[d], j)
    tiled_stencil_kernel[(16, 16), (16, 16), stream_boundary](u, kappa, source, dest, dt, 1.0 / (dx * dx), 1 if overlapped else 2)
    stream_boundary.synchronize()
    if overlapped:
        stream_interior.synchronize()
    return dest, u


def run_solver(ctx, mode, state):
    u = cuda.device_array_like(state['u'])
    u.copy_to_device(state['u'])
    dest = cuda.device_array_like(u)
    buffers = make_buffers()
    s1 = cuda.stream()
    s2 = cuda.stream()
    cart = ctx['cart']
    snapshots = {}
    cart.Barrier()
    snapshots['0'] = energy(ctx, u)
    cart.Barrier()
    start = MPI.Wtime()
    for t in range(1, STENCIL_STEPS + 1):
        u, dest = step(ctx, u, dest, state['kappa'], state['source'], state['dt'], state['dx'], buffers, s1, s2, mode == 'overlapped')
        if t in (100, 200, 300):
            snapshots[str(t)] = energy(ctx, u)
    cuda.synchronize()
    cart.Barrier()
    ms = cart.allreduce((MPI.Wtime() - start) * 1000.0, op=MPI.MAX)
    return u, ms, snapshots


def benchmark_shm_strides(ctx):
    node = ctx['node']
    shm = ctx['shm']
    results = {}
    host_shm_accumulate(shm, 0, 1)
    node.Barrier()
    for stride in (1, 8, 16):
        if ctx['node_rank'] == 0:
            shm.fill(0)
        ctx['win'].Sync()
        node.Barrier()
        slot = ctx['node_rank'] * stride
        start = MPI.Wtime()
        host_shm_accumulate(shm, slot, SHM_ITERS)
        node.Barrier()
        elapsed = node.allreduce((MPI.Wtime() - start) * 1000.0, op=MPI.MAX)
        ctx['win'].Sync()
        node.Barrier()
        values = [int(shm[i * stride]) for i in range(4)] if ctx['node_rank'] == 0 else None
        values = node.bcast(values, root=0)
        if values != [SHM_ITERS] * 4:
            raise RuntimeError(f'Shared-memory result is invalid for stride={stride}: {values}')
        results[str(stride)] = dict(ms=elapsed, spacing_bytes=8 * stride, values=values)
    return results


def audit_gpu():
    cuda.synchronize()
    asm = tiled_stencil_kernel.inspect_asm()
    if isinstance(asm, dict):
        asm = '\n'.join(str(value) for value in asm.values())
    pattern = lambda name: len(re.findall(r'(?<![A-Za-z0-9_.])' + re.escape(name) + r'(?=[;\s.])', asm))
    return dict(st_shared_f64=pattern('st.shared.f64'), ld_shared_f64=pattern('ld.shared.f64'), bar_sync=pattern('bar.sync'), shared_store_all=len(re.findall(r'\bst\.shared\.', asm)), shared_load_all=len(re.findall(r'\bld\.shared\.', asm)), ptx_chars=len(asm))


def audit_cpu():
    llvm = '\n'.join(str(host_shm_accumulate.inspect_llvm(sig)) for sig in host_shm_accumulate.signatures)
    return dict(load_count=len(re.findall(r'\bload\b', llvm)), store_count=len(re.findall(r'\bstore\b', llvm)), memory_barrier_count=llvm.count('asm sideeffect'), volatile_count=len(re.findall(r'\bvolatile\b', llvm)), llvm_chars=len(llvm), loop_audit_note='Inspect the emitted LLVM to verify that no load/store hoisting occurred')


def main():
    username = sys.argv[1] if len(sys.argv) > 1 else os.environ.get('GITHUB_USERNAME', '').strip()
    if not username:
        raise RuntimeError('Pass your real GitHub username as the first argument')
    ctx = init_hybrid_context(username)
    try:
        state = initialize_fields(ctx)
        serial, serial_ms, serial_energy = run_solver(ctx, 'serialized', state)
        overlap, overlap_ms, overlap_energy = run_solver(ctx, 'overlapped', state)
        d1 = serial.copy_to_host()[1:257, 1:257]
        d2 = overlap.copy_to_host()[1:257, 1:257]
        local_diff = float(np.max(np.abs(d1 - d2)))
        max_diff = ctx['cart'].allreduce(local_diff, op=MPI.MAX)
        if max_diff >= 1e-12:
            raise RuntimeError(f'Serialized/overlapped mismatch: {max_diff}')
        shm = benchmark_shm_strides(ctx)
        gpu_audit = audit_gpu()
        cpu_audit = audit_cpu()
        gathered_gpu = ctx['comm'].gather(gpu_audit, root=0)
        gathered_cpu = ctx['comm'].gather(cpu_audit, root=0)
        manifest = dict(github_username=username, instance_seed=ctx['instance_seed'], gpu_uuid=ctx['gpu_uuid'], cpu_signature=ctx['cpu_signature'], cpu_cache_line_bytes=ctx['cache_line_size'], gpu_name=cuda.get_current_device().name.decode() if isinstance(cuda.get_current_device().name, bytes) else str(cuda.get_current_device().name), mpi_ranks=4, mpi_grid=[2, 2], global_domain=[512, 512], local_padded_domain=[258, 258], stencil_steps=STENCIL_STEPS, shm_iterations=SHM_ITERS, kappa_max=state['kappa_max'], dx=state['dx'], dt=state['dt'], dt_max=state['dt_max'], cfl_stable=bool(state['dt'] <= state['dt_max']), gpu_serialized_ms=serial_ms, gpu_overlapped_ms=overlap_ms, gpu_speedup=serial_ms / overlap_ms if overlap_ms else None, max_diff=max_diff, parity_passed=bool(max_diff < 1e-12), global_energy_serialized=serial_energy, global_energy_overlapped=overlap_energy, host_shm_strides=shm, false_sharing_slowdown=shm['1']['ms'] / shm['8']['ms'] if shm['8']['ms'] else None, stride_16_over_8=shm['16']['ms'] / shm['8']['ms'] if shm['8']['ms'] else None, ptx_audit_by_rank=gathered_gpu, llvm_audit_by_rank=gathered_cpu)
        if ctx['rank'] == 0:
            with open('hybrid_lab_manifest.json', 'w', encoding='utf-8') as f:
                json.dump(manifest, f, indent=2)
            print(json.dumps(manifest, indent=2), flush=True)
    finally:
        ctx['comm'].Barrier()
        ctx['win'].Free()
        ctx['node'].Free()
        ctx['cart'].Free()


if __name__ == '__main__':
    main()
