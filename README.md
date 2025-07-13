# CIFAR-10 Parallel Training Project

This project aims to analyze the performance of various deep learning models on the **CIFAR-10** dataset by comparing different **parallelization strategies**. All experiments were conducted on the **ARF Supercomputer** using distributed training techniques.

## Infrastructure

- **System**: ARF Supercomputer  
- **Job Scheduler**: SLURM  
- **GPUs**: NVIDIA A100 and V100  
- **CPUs**: 10-core and 20-core configurations  
- **Partitions**: `akya-cuda`, `barbun-cuda`  
- **Distributed Training**: PyTorch Lightning with FSDP / DDP support  

## Experimental Settings

The following combinations of training parameters were tested to evaluate performance under different scenarios:

- **Batch Sizes**: 32, 128, 512  
- **Optimizers**: SGD, Adam, RMSprop  
- **Learning Rates**: 0.01, 0.001  
- **Schedulers**: None, StepLR, CosineAnnealingLR  
- **Number of GPUs**: 2, 4  
- **Epochs**: 10  

## Project Goals

- Evaluate training speed and GPU/CPU utilization across parallel setups  
- Compare optimizer and scheduler impacts under different batch sizes  
- Demonstrate distributed training using PyTorch Lightning (FSDP and DDP)  
- Benchmark training time, accuracy, and scalability on a national HPC system  

## Results

Detailed experiment logs, training curves, and metric-based comparisons can be found in the `results/` directory. These reports include:

- Training/validation accuracy and loss plots  
- Resource usage summaries  
- Per-configuration result summaries  

All SLURM job submission scripts used for parallel training experiments are available in the scripts/ folder. These scripts can be modified to adjust GPU count, partition, model settings, and hyperparameters.
