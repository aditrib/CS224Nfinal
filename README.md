# CS224Nfinal
Default Final Project


G6 Dark Arts Model

Iterative
10 Epochs (8 completed, then stopped)
Conv CLFs
LR 2e-4 
Lambda 0.5 scheduling 
Adamax for all:
    sst_optimizer = optim.Adamax(model.parameters(), lr=sst_lr, weight_decay=1e-5)   # SST needs larger learning rate
    para_optimizer = optim.Adamax(model.parameters(), lr=para_lr, weight_decay=1e-3)
    sts_optimizer = optim.Adamax(model.parameters(), lr=sts_lr, weight_decay=5e-3)
