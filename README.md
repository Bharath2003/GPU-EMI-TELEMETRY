# GPU-EMI-TELEMETRY
GPU-Accelerated EMI Filtering for High-Speed Automotive Infotainment Streams


# GPU-Accelerated EMI Filtering for High-Speed Automotive Telemetry

## 📌 Overview
Modern electric and autonomous vehicles generate millions of telemetry samples per second. [cite_start]High-power components like traction inverters create broadband switching noise and high-energy electromagnetic interference (EMI)[cite: 24]. 

[cite_start]This project implements a high-throughput, parallelized GPU processing engine written in **CUDA C++** for real-time anomaly detection and "blind" filtering[cite: 11, 26]. [cite_start]It shifts digital signal processing (DSP) workloads from a linear CPU bottleneck to a fully parallel SIMT execution model, achieving a **31.64x speedup** over traditional sequential methods[cite: 14, 19].

## 🚀 Key Features
* [cite_start]**Zero-Copy DMA Transport:** Utilizes pinned memory (`cudaHostAlloc`) to bypass CPU staging buffers, reducing PCIe data transport latency by 6.14x[cite: 49, 101, 104].
* [cite_start]**$O(1)$ Parallel Thresholding:** Implements a custom frequency-gated CUDA kernel that evaluates and suppresses multiple concurrent fault frequencies simultaneously, independently of the number of anomalies[cite: 51, 84].
* [cite_start]**In-Place VRAM Optimization:** Uses shared memory pointers for forward and inverse FFTs, cutting the active VRAM footprint in half (from ~33.5 MB to 16.77 MB)[cite: 106, 107].

## 🛠️ Architecture Pipeline
[cite_start]The engine executes a 4-stage edge-computing pipeline[cite: 45]:
1. [cite_start]**Data Ingestion:** Host CPU ingests raw telemetry arrays (e.g., 4.19M points)[cite: 48].
2. [cite_start]**Zero-Copy Transport:** Data is pulled directly into the GPU via Direct Memory Access (DMA)[cite: 59].
3. [cite_start]**GPU DSP Execution:** * Forward Transform using `cuFFT` (Real-to-Complex)[cite: 61].
   * [cite_start]Parallel Anomaly Suppression via custom SIMT dynamic thresholding kernel[cite: 62].
   * [cite_start]Inverse Transform using `cuFFT` (Complex-to-Real)[cite: 64].
4. [cite_start]**Signal Extraction:** Host extracts the normalized base signal utilizing `cuBLAS` scaling[cite: 65, 66].

## 📊 Benchmarks & Results
[cite_start]Tested on a 4.19 million point dataset with three simultaneous high-energy mechanical/electrical faults[cite: 116, 117]:
* [cite_start]**GPU Latency:** 4.451 ms [cite: 122]
* [cite_start]**CPU Latency (FFTW3):** 140.807 ms [cite: 120]
* [cite_start]**True Scientific Speedup:** 31.64x [cite: 122]
* [cite_start]**Signal Recovery:** Complete anomaly deletion while perfectly preserving the ground-truth 60Hz mechanical baseline[cite: 129, 130].

## 💻 Dependencies
To compile and run this project, you will need:
* [cite_start]NVIDIA CUDA Toolkit (v12.5+) [cite: 118]
* [cite_start]`cuFFT` Library [cite: 94]
* [cite_start]`cuBLAS` Library [cite: 95]
* [cite_start]`FFTW3` (for CPU baseline benchmarking) [cite: 120]

## ⚙️ How to Run
1. Generate the synthetic telemetry arrays (creates binary files):
   ```bash
   python generate_telemetry.py
