# CIFAR-100 Parallel Training Project
This project was developed as part of the Artificial Intelligence Specialization Program, utilizing TÜBİTAK's High Performance Computing (HPC) infrastructure.

I analyzed the performance of various deep learning configurations on the CIFAR-100 dataset using different parallelization strategies. All experiments were run on the ARF Supercomputer using distributed training techniques.

## Infrastructure
- **System**: ARF Supercomputer  
- **Job Scheduler**: SLURM  
- **GPUs**: NVIDIA A100 and V100  
- **CPUs**: 10-core and 20-core configurations  
- **Partitions**: `akya-cuda`, `barbun-cuda`  
- **Distributed Training**: PyTorch Lightning with FSDP / DDP support  

## Experimental Settings
All experiments used ResNet50 as the base model. The following parameter combinations were tested:
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

All experiment results are located in the results/ folder and include:
- Resource usage (CPU/GPU utilization, memory)
- Tabular comparisons of all configurations

All SLURM job submission scripts used for parallel training experiments are available in the scripts/ folder. These scripts can be modified to adjust GPU count, partition, model settings, and hyperparameters.
