# CPU-side changes required for forward-only range search

These changes complement `no_flip.patch` (GPU kernel) and `GPUCompute_Ada.h`.

---

## Why the CPU side needs changes

The original VanitySearch/KeyHunt-Cuda code uses a **centred group**: each group of
`GRP_SIZE` keys is placed symmetrically around `startKey`.  The CPU stores thread
starting-points such that:

```
Thread 0:  startKey = rangeStart
           Group covers [rangeStart - GRP_SIZE/2, rangeStart + GRP_SIZE/2 - 1]
                                   ↑ OUTSIDE range!
```

After removing the backward direction the kernel checks only:
```
Thread 0:  startKey = rangeStart
           Group covers [rangeStart, rangeStart + GRP_SIZE - 1]  ✓
```

Two changes are needed in `GPUEngine.cpp`.

---

## Change 1 — Extend the Gx/Gy precomputed table (GPUEngine.cpp)

### What it is
`Gx` and `Gy` are GPU device arrays holding the x- and y-coordinates of
`G, 2G, 3G, …` up to `HSIZE*G` (where `HSIZE = GRP_SIZE/2 - 1`).

The forward-only kernel needs entries up to `GRP_SIZE*G`, which is currently
stored only as `_2Gnx`/`_2Gny`.

### What to change

Find the loop that fills `Gx` / `Gy` (looks like):

```cpp
// ORIGINAL — fills HSIZE+1 entries
vector<uint64_t> hGx, hGy;
Int pk;
pk.SetInt32(1);
Point P = secp->ComputePublicKey(&pk);
for (int i = 0; i < HSIZE + 2; i++) {
    hGx.insert(..., P.x ...);
    hGy.insert(..., P.y ...);
    P = secp->NextKey(P);    // P += G
}
```

Change the loop bound from `HSIZE + 2` to `GRP_SIZE + 1`:

```cpp
// PATCHED — fills GRP_SIZE+1 entries (double the original)
for (int i = 0; i < GRP_SIZE + 1; i++) {
    hGx.insert(..., P.x ...);
    hGy.insert(..., P.y ...);
    P = secp->NextKey(P);
}
```

The CUDA `cudaMemcpyToSymbol` call for `Gx` / `Gy` must also copy the larger array.

---

## Change 2 — Fix the incr → private key decoding (GPUEngine.cpp)

### What it is
When the GPU finds a match it writes an `incr` value to the result buffer.
The CPU uses `incr` to recover the actual private key:

```cpp
// ORIGINAL (centred group)
Int pk = threadStartKey;
pk.Add(incr - GRP_SIZE / 2);   // ← subtracts half when incr < GRP_SIZE/2
```

With the forward-only kernel `incr` is always 0 ≤ incr < GRP_SIZE, and:

```cpp
// PATCHED (forward-only)
Int pk = threadStartKey;
pk.Add(incr);                   // ← always positive, no centre offset
```

### Where to find it

Search `GPUEngine.cpp` for the result-parsing loop.  It will contain code
similar to:

```cpp
// Report found key
int incr  = (int)(out[pos * ITEM_SIZE_A32 + 2] >> 16);
Int found = startKey[tid];
found.Add(incr - GRP_SIZE / 2);   // ← change this line
```

Change to:

```cpp
Int found = startKey[tid];
found.Add(incr);                   // ← no centre offset
```

---

## Change 3 — Thread starting-key initialisation (GPUEngine.cpp)

### What it is
The original code initialises `startKey[tid]` as the **centre** of the group:

```cpp
// ORIGINAL
Int start = rangeStart;
for (int tid = 0; tid < nbThread; tid++) {
    startKey[tid] = start;         // centre = rangeStart + tid*GRP_SIZE
    start.Add(GRP_SIZE);
}
```

Because the group is now forward-only (starts at `startKey`, ends at
`startKey + GRP_SIZE - 1`), this initialisation is already correct — no change
needed.  The first thread checks `[rangeStart, rangeStart + GRP_SIZE - 1]`,
the second checks `[rangeStart + GRP_SIZE, rangeStart + 2*GRP_SIZE - 1]`, etc.

---

## Summary of changes

| File | Change | Lines affected |
|---|---|---|
| `GPU/GPUCompute.h` | Apply `no_flip.patch` | ~50 lines across 4 functions |
| `GPUEngine.cpp` | Extend Gx/Gy table fill loop bound: `HSIZE+2` → `GRP_SIZE+1` | ~5 lines |
| `GPUEngine.cpp` | Change incr decode: `incr - GRP_SIZE/2` → `incr` | ~2 lines |

---

## Performance impact

| Metric | Before (bidirectional) | After (forward-only) |
|---|---|---|
| Batch inversion size | `GRP_SIZE/2 + 1` | `GRP_SIZE + 1` |
| Extra ModMult in `_ModInvGrouped` | baseline | +~3×(GRP_SIZE/2) ≈ +192 for GRP_SIZE=128 |
| Total compute overhead | baseline | ~+10–15% |
| Keys outside range checked | up to `GRP_SIZE/2` per boundary | **0** |
| Risk of negative-incr key report | yes | **eliminated** |
| Gx/Gy GPU table size | `(GRP_SIZE/2 + 1) × 64 bytes` | `(GRP_SIZE + 1) × 64 bytes` |

The ~10–15% extra batch-inversion cost is more than recovered by the other Ada
optimisations (L2 bloom persistence, sm_89 compile target, CUDA Graphs).
