# KeyHunt-Cuda — Optimized Makefile
# Targets: Ada Lovelace (sm_89, RTX 40xx) and Blackwell Consumer (sm_120, RTX 50xx)
#
# Usage:
#   RTX 40xx:  make gpu=1 CCAP=89  all
#   RTX 50xx:  make gpu=1 CCAP=120 all
#   Multi-arch: make gpu=1 CCAP=MULTI all
#
# Requirements:
#   sm_89  (Ada):      CUDA 11.8+  (CUDA 12.x strongly recommended)
#   sm_120 (Blackwell): CUDA 12.8+, Driver 570+

SRC      = Base58.cpp IntGroup.cpp main.cpp Random.cpp \
           Timer.cpp Int.cpp IntMod.cpp Point.cpp SECP256K1.cpp \
           GPU/GPUGenerate.cpp hash/ripemd160.cpp hash/sha256.cpp \
           hash/sha512.cpp hash/ripemd160_sse.cpp hash/sha256_sse.cpp \
           Bech32.cpp Bloom.cpp

OBJDIR   = obj

# ─── Compiler ────────────────────────────────────────────────────────────────
CUDA     ?= /usr/local/cuda
CXXCUDA  ?= /usr/bin/g++
NVCC      = $(CUDA)/bin/nvcc

# Host flags — enable AVX2/AVX-512 for CPU-side bloom filter ops
CXXFLAGS = -m64 -mssse3 -Wno-unused-result -Wno-write-strings \
           -O3 -march=native -funroll-loops -fomit-frame-pointer

# ─── GPU / CCAP Selection ────────────────────────────────────────────────────
CCAP     ?= 89

ifeq ($(CCAP),MULTI)
  # Fat binary: Ampere + Ada + Blackwell consumer
  GENCODE  = -gencode=arch=compute_86,code=sm_86  \
             -gencode=arch=compute_89,code=sm_89  \
             -gencode=arch=compute_120,code=sm_120 \
             -gencode=arch=compute_120,code=compute_120
  $(info [INFO] Multi-arch fat binary: sm_86 + sm_89 + sm_120)
else ifeq ($(CCAP),120)
  # Blackwell consumer — requires CUDA 12.8+
  GENCODE  = -gencode=arch=compute_120,code=sm_120 \
             -gencode=arch=compute_120,code=compute_120
  $(info [INFO] Target: Blackwell consumer sm_120 — ensure CUDA >= 12.8 and Driver >= 570)
else ifeq ($(CCAP),89)
  # Ada Lovelace — requires CUDA 11.8+ (12.x recommended)
  GENCODE  = -gencode=arch=compute_89,code=sm_89  \
             -gencode=arch=compute_89,code=compute_89
  $(info [INFO] Target: Ada Lovelace sm_89 — ensure CUDA >= 11.8)
else
  # Fallback: single-arch as specified
  GENCODE  = -gencode=arch=compute_$(CCAP),code=sm_$(CCAP)
  $(info [INFO] Target: sm_$(CCAP))
endif

# ─── NVCC Flags ──────────────────────────────────────────────────────────────
#
# Key changes vs. the default VanitySearch Makefile:
#   -O3               : more aggressive than -O2
#   -use_fast_math    : ftz + fast div/sqrt — safe for hash/key workloads
#   --extra-device-vectorization : enables extra loop vectorization passes
#   Second gencode    : embeds PTX for forward JIT compatibility
#   -maxrregcount=0   : let compiler decide; verify via --ptxas-options=-v
#                       (if register count > 64, set to 64 to raise occupancy)
#
NVCC_FLAGS_COMMON = \
  -maxrregcount=0              \
  --ptxas-options=-v           \
  --compile                    \
  --compiler-options -fPIC     \
  -ccbin $(CXXCUDA)            \
  -m64                         \
  -I$(CUDA)/include            \
  -use_fast_math               \
  --extra-device-vectorization \
  $(GENCODE)

NVCC_FLAGS_RELEASE = $(NVCC_FLAGS_COMMON) -O3
NVCC_FLAGS_DEBUG   = $(NVCC_FLAGS_COMMON) -G -g -O0

# ─── Object lists ────────────────────────────────────────────────────────────
OBJS     = $(addprefix $(OBJDIR)/,$(SRC:.cpp=.o))

ifeq ($(gpu),1)
  CXXFLAGS  += -DWITHGPU
  LFLAGS    = -L$(CUDA)/lib64 -lcudart
  GPU_OBJ   = $(OBJDIR)/GPU/GPUEngine.o
  # Use the Ada/Blackwell-optimised engine if available
  GPU_SRC   = GPU/GPUEngine_Ada.cu
  ifeq (,$(wildcard $(GPU_SRC)))
    GPU_SRC = GPU/GPUEngine.cu
  endif
endif

# ─── Targets ─────────────────────────────────────────────────────────────────
all: keyhunt

keyhunt: $(OBJS) $(GPU_OBJ)
	$(CXXCUDA) $(CXXFLAGS) -o keyhunt $^ $(LFLAGS)

# Host objects
$(OBJDIR)/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXXCUDA) $(CXXFLAGS) -o $@ -c $<

# GPU object — release build
$(OBJDIR)/GPU/GPUEngine.o: $(GPU_SRC)
	@mkdir -p $(OBJDIR)/GPU
	$(NVCC) $(NVCC_FLAGS_RELEASE) -o $@ $<

# GPU object — debug build
$(OBJDIR)/GPU/GPUEngine_debug.o: $(GPU_SRC)
	@mkdir -p $(OBJDIR)/GPU
	$(NVCC) $(NVCC_FLAGS_DEBUG) -o $@ $<

clean:
	rm -rf $(OBJDIR) keyhunt

.PHONY: all clean
