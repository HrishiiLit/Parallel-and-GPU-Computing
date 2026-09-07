Matrix Multiplication using Sequential, OpenMP, MPI and CUDA

A comparative parallel computing project implementing 4000 × 4000 matrix multiplication using four different computing models:

Sequential C — Single CPU execution
OpenMP — Shared-memory CPU parallelism
MPI — Distributed-memory parallelism across four Ubuntu VMs
CUDA — GPU parallelism using an NVIDIA GPU

The same mathematical problem is solved in all four implementations, allowing their execution times and speedups to be compared directly.

📌 Overview

Matrix multiplication is a computationally intensive operation that is well suited for demonstrating different parallel programming models.

For this experiment:

Matrix A = 4000 × 4000
Matrix B = 4000 × 4000
Every element of A and B is initialized to 1.0
C = A × B

Since every element contains the sum of 4000 products of 1 × 1:

C[i][j] = 4000.00


Therefore, the expected verification result is:

C[0][0] = 4000.00

🧠 Computing Models
1. Sequential

The baseline implementation performs matrix multiplication using a single CPU execution flow.

A × B
 ↓
Single CPU
 ↓
C

2. OpenMP

OpenMP divides the outer matrix rows among multiple CPU threads while sharing the same memory.

             ┌── Thread 1
             ├── Thread 2
A × B ───────┼── Thread 3
             ├── ...
             └── Thread 8
                  ↓
                  C


The reference experiment uses 8 OpenMP threads.

3. MPI

MPI distributes matrix rows across independent processes running on multiple Ubuntu virtual machines.

                    Matrix A
                       │
                 MPI_Scatter
                       │
       ┌───────────────┼───────────────┐
       ↓               ↓               ↓
    Rank 0           Rank 1          Rank 2 ... Rank 3
   1000 rows        1000 rows        1000 rows
       │               │               │
       └───────────────┼───────────────┘
                   MPI_Gather
                       ↓
                  Complete C


The reference configuration uses:

1 Master VM
3 Worker VMs
4 MPI processes
1000 rows per process
4. CUDA

CUDA assigns GPU threads to output matrix elements.

CPU
 │
 ├── Copy A → GPU
 ├── Copy B → GPU
 │
 ↓
GPU Kernel
 │
 ├── Thousands of blocks
 └── Millions of logical threads
 │
 ↓
Copy C → CPU


The reference implementation uses:

16 × 16 threads per block
250 × 250 blocks
62,500 total blocks
16,000,000 logical CUDA thread instances
🗂️ Project Structure
matrix-multiplication-parallel/
│
├── sequential/
│   ├── matrix_sequential.c
│   └── README.md
│
├── openmp/
│   ├── matrix_openmp.c
│   └── README.md
│
├── mpi/
│   ├── matrix_mpi.c
│   ├── hosts.example
│   └── README.md
│
├── cuda/
│   ├── matrix_cuda.cu
│   └── README.md
│
├── results/
│   ├── sequential.png
│   ├── openmp.png
│   ├── mpi.png
│   ├── cuda.png
│   └── performance-comparison.png
│
├── docs/
│   ├── architecture.md
│   └── screenshots/
│
├── .gitignore
└── README.md

⚙️ Requirements
General
Linux / Ubuntu environment
GCC
build-essential
Git
Sequential
GCC
OpenMP
GCC with OpenMP support
MPI
Ubuntu
Open MPI
OpenSSH
Multiple network-connected Ubuntu machines/VMs
CUDA
NVIDIA CUDA-capable GPU
NVIDIA driver
CUDA Toolkit
nvcc
🚀 Part A — Sequential
Environment

The reference experiment uses Ubuntu running inside WSL2 on Windows.

Verify WSL from Windows PowerShell:

wsl --status
wsl -l -v


Start Ubuntu:

wsl


Inside Ubuntu:

sudo apt update
sudo apt install build-essential -y
gcc --version

Build
cd sequential

gcc -O2 matrix_sequential.c -o matrix_sequential

Run
./matrix_sequential


Expected output:

Sequential Matrix Multiplication Completed
Matrix Size = 4000 x 4000
Execution Time = 244.120000 seconds
Verification C[0][0] = 4000.00

🚀 Part B — OpenMP
Check CPU Availability
nproc


Set the number of OpenMP threads:

export OMP_NUM_THREADS=8


Verify:

echo $OMP_NUM_THREADS

Build
cd openmp

gcc -O2 -fopenmp matrix_openmp.c -o matrix_openmp

Run
./matrix_openmp


Expected output:

OpenMP Matrix Multiplication Completed
Matrix Size = 4000 x 4000
Number of Threads Used = 8
Execution Time = 30.830434 seconds
Verification C[0][0] = 4000.00


The reference experiment achieved approximately:

7.92× speedup


over the sequential implementation.

🚀 Part C — MPI

The MPI experiment uses four Ubuntu VMs.

Node	Hostname	MPI Rank
Master	master	0
Worker 1	worker1	1
Worker 2	worker2	2
Worker 3	worker3	3

The IP addresses used in the original lab are environment-specific. Do not commit private/local IP addresses to the repository. Use hosts.example as a template instead.

Install MPI

Run on every VM:

sudo apt update
sudo apt install openmpi-bin libopenmpi-dev -y


Install SSH:

sudo apt install openssh-server -y
sudo systemctl enable --now ssh


Verify:

mpicc --version
mpirun --version

Configure SSH

On the Master:

ssh-keygen -t rsa


Copy the public key:

ssh-copy-id worker1
ssh-copy-id worker2
ssh-copy-id worker3


Test:

ssh worker1 hostname
ssh worker2 hostname
ssh worker3 hostname

Hostfile

Create a hostfile based on your own VM configuration.

Example:

master slots=1
worker1 slots=1
worker2 slots=1
worker3 slots=1

Build
mpicc -O2 matrix_mpi.c -o matrix_mpi


Copy the executable to the workers:

scp matrix_mpi worker1:~/matrix_mpi
scp matrix_mpi worker2:~/matrix_mpi
scp matrix_mpi worker3:~/matrix_mpi

Run
mpirun -np 4 --hostfile hosts sh -c '$HOME/matrix_mpi'


Expected result:

MPI Matrix Multiplication Completed
Matrix Size = 4000 x 4000
Number of MPI Processes = 4
Execution Time = 92.979510 seconds
Verification C[0][0] = 4000.00


Reference speedup:

2.63×

🚀 Part D — CUDA
Verify NVIDIA GPU
nvidia-smi


Verify CUDA:

nvcc --version

Build
cd cuda

nvcc -O2 matrix_cuda.cu -o matrix_cuda

Run
./matrix_cuda


Expected output:

CUDA Matrix Multiplication Completed
Matrix Size = 4000 x 4000
Grid Size = 250 x 250 blocks
Block Size = 16 x 16 threads
Kernel Execution Time = 0.146443 seconds
Total CUDA Phase Time = 0.165004 seconds
Verification C[0][0] = 4000.00


The measured total CUDA phase achieved approximately:

1479.48× speedup


over the sequential baseline.

📊 Performance Comparison

All four implementations produce:

C[0][0] = 4000.00

Implementation	Computing Model	Resources	Execution Time	Speedup
Sequential	Single CPU	1 CPU execution flow	244.120000 s	1.00×
OpenMP	Shared memory	8 CPU threads	30.830434 s	7.92×
MPI	Distributed memory	4 processes / 4 VMs	92.979510 s	2.63×
CUDA	GPU parallelism	NVIDIA RTX 4500 Ada	0.165004 s	1479.48×
Speedup Formula
Speedup = Sequential Execution Time / Parallel Execution Time


For example, OpenMP:

244.120000 / 30.830434 ≈ 7.92×

📈 Results

The experiment demonstrates the performance differences between several parallel programming models.

Sequential

The sequential implementation provides the baseline execution time.

OpenMP

OpenMP significantly reduces execution time by distributing work among multiple CPU threads. With eight threads, the reference experiment achieved approximately 7.92× speedup.

MPI

MPI allows computation to be distributed across multiple machines. However, communication, data distribution, synchronization and VM/network overhead reduce the benefit compared with local shared-memory parallelism.

The reference experiment achieved approximately 2.63× speedup using four VMs.

CUDA

CUDA provides the largest performance improvement for this workload. The GPU executes a very large number of threads in parallel.

The reference experiment achieved approximately 1479.48× speedup using the total CUDA phase time.

Performance results depend heavily on hardware, compiler versions, CPU configuration, VM configuration, GPU model, memory bandwidth and system load. The values above are the recorded results from the reference experiment and should not be treated as universal benchmarks.

🔬 Important CUDA Timing Note

Two CUDA timings are reported:

Kernel-only time       = 0.146443 s
Total CUDA phase time  = 0.165004 s


The total CUDA phase time includes:

Host-to-device transfer
GPU kernel execution
Device-to-host transfer

Therefore, the performance comparison uses:

244.120000 / 0.165004 ≈ 1479.48×


This provides a more complete measurement of the CUDA portion of the application.

🧪 Verification

The matrices are initialized as:

A[i][j] = 1.0
B[i][j] = 1.0


Therefore:

C[i][j] = Σ A[i][k] × B[k][j]

         = 1 + 1 + ... + 1
                    ↑
                 4000 terms

         = 4000


The implementations verify the result using:

C[0][0]


Expected:

4000.00

🛠️ Troubleshooting
WSL is not recognized

From PowerShell:

wsl --status
wsl -l -v


Make sure an Ubuntu distribution is installed and running under WSL2.

GCC is not found
sudo apt update
sudo apt install build-essential -y

OpenMP compilation fails

Make sure -fopenmp is included:

gcc -O2 -fopenmp matrix_openmp.c -o matrix_openmp

OpenMP uses fewer threads

Check:

nproc
echo $OMP_NUM_THREADS

MPI cannot connect to workers

Check:

ping worker1
ping worker2
ping worker3


Then test SSH:

ssh worker1 hostname

MPI asks for a password

Configure passwordless SSH using:

ssh-copy-id worker1
ssh-copy-id worker2
ssh-copy-id worker3

CUDA compiler is not found

Check:

nvcc --version


Verify that the CUDA Toolkit is installed and that nvcc is available in your PATH.

nvidia-smi cannot detect the GPU

Verify that the NVIDIA driver is installed correctly and that the operating system can see the GPU.

📸 Experiment Evidence

Screenshots demonstrating the experiment are included in the results/ directory.

Recommended evidence:

WSL2 verification
GCC installation/version
Sequential compilation and output
OpenMP thread configuration
OpenMP execution
htop showing CPU utilization
MPI network connectivity
Passwordless SSH
MPI ranks running across nodes
nvidia-smi
CUDA compilation
CUDA execution output
Final performance comparison
🎯 Learning Objectives

This project demonstrates:

Sequential matrix multiplication
Shared-memory parallelism
OpenMP thread parallelism
Distributed-memory parallelism
MPI communication
GPU programming with CUDA
CUDA kernel execution
CPU/GPU memory transfers
Performance measurement
Parallel speedup analysis
Comparison of computing architectures
📚 Technologies Used
C
OpenMP
MPI / Open MPI
CUDA C/C++
GCC
NVIDIA CUDA Toolkit
Ubuntu
WSL2
VMware Workstation
OpenSSH
👨‍💻 Author

Your Name

Parallel Computing Laboratory Project

⭐ Acknowledgement

This repository was created as part of a parallel computing laboratory experiment to study and compare sequential, shared-memory, distributed-memory and GPU-based matrix multiplication.
