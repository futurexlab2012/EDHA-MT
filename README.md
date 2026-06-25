# EDHA-MT
EDHA-MT Spiking Sparse Computing Event-Driven Framework


The open-source code of the paper "EDHA-MT: An Event-Driven Highly Accurate SNN Simulation Framework with Multi-threads Parallel Computing"


Usage:

SNNMachineRun1 snn1 = new SNNMachineRun1(...);\
SNNMachineRun2 snn2 = new SNNMachineRun2(...);

snn1.start();\
snn2.start();

snn1.join();\
snn2.join();
