# CIFAR10 Paralel Eğitim Projesi 

Bu proje, CIFAR10 veri seti üzerinde çeşitli paralelleştirme stratejilerini karşılaştırarak farklı derin öğrenme modellerinin performansını analiz etmeyi amaçlamaktadır.
Eğitimler, ARF süper bilgisayarı üzerinde gerçekleştirilmiştir.


Kullanılan Altyapı

- Sistem: ARF Süper Bilgisayarı
- Yönetim Sistemi: SLURM
- GPU: NVIDIA A100 ve V100
- CPU: 10 ve 20
- Partition: akya-cuda, barbun-cuda
- Dağıtık Eğitim: PyTorch Lightning + FSDP / DDP

# Deneysel Ayarlar

Aşağıdaki parametre kombinasyonları test edilmiştir:

- Batch Size: 32, 128, 512
- Optimizer: SGD, Adam, RMSprop
- Learning Rate: 0.01, 0.001
- Scheduler: None, StepLR, CosineAnnealingLR
- GPU Sayısı: 2, 4
- Epoch: 10 

## Detaylı sonuç raporu results klasörü altında bulunmaktadır.
