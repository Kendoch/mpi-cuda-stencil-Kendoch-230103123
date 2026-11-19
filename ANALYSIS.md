# Cache Line & False Sharing Mechanics — Analysis

### Q1. MESI Protocol State Transitions

In stride = 1, Core 0 writes to `buffer[0]`, and Core 1 writes to `buffer[1]`.

These two values are in the same cache line.

First, both cores may have the line in the Shared (S) state. When Core 0 writes, it gets ownership of the line. Its state changes to Modified (M), while Core 1's copy becomes Invalid (I).

When Core 1 wants to write, it sends a Read-For-Ownership (RFO) request. Core 0 gives up ownership, and Core 1 gets the line in Modified (M) state.

This happens repeatedly, even though the cores update different values.

The store buffer may have to wait for ownership before the write becomes globally visible. This makes execution slower.

### Q2. Quantitative Slowdown & Coherency Overhead

My benchmark used 4 threads and 60,000,000 updates per thread.

| Layout | Time (ns) | ns/update |
|---|---:|---:|
| Stride 1 | 143,871,417 | 0.599464 |
| Stride 4 | 141,526,833 | 0.589695 |
| Stride 8 | 138,386,542 | 0.576611 |
| Stride 16 | 138,854,250 | 0.578559 |
| Private | 294,208 | 0.001226 |

The slowdown factor is:

143,871,417 / 138,386,542 = **1.0396×**

This means stride 1 was about 3.96% slower than stride 8.

The private version was approximately 489 times faster than stride 1.

This shows that memory updates are more expensive than keeping a value in a CPU register.

### Q3. L2 Spatial Prefetcher & 128-Byte Pairing

My measured results are:

- Stride 8: 138,386,542 ns
- Stride 16: 138,854,250 ns

Stride 8 was slightly faster than stride 16, by about 0.34%.

My CPU is ARM64, and its reported cache-line size is 128 bytes.

This is important because stride 8 means 64-byte spacing, so two counters can still share one 128-byte cache line.

Stride 16 means 128-byte spacing, so the counters occupy different cache lines when properly aligned.

Some processors also use adjacent-line prefetching, which can affect memory traffic and performance.

In my test, stride 16 did not improve performance. The small difference may be caused by system noise or other CPU effects.

### Q4. Compiler Register Promotion vs. Memory Stores

An optimizing compiler can move a frequently used value into a CPU register instead of reading and writing memory in every loop iteration.

This is called register promotion.

In my code, I used an LLVM compiler memory barrier to prevent the compiler from removing or combining memory operations.

My LLVM audit showed:

- Native i64 stores: 1
- Native i64 loads: 4
- Memory barriers: 1
- Native compilation: True

The audit also reported `llvm_loop_store_detected = True`.

This confirms that the generated LLVM IR contains native memory operations and a compiler barrier.

The exact machine-code behavior would need assembly