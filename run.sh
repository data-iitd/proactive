device=0
data=data/Breakfast/
batch=4
n_head=4
n_layers=4
d_model=64
d_inner=256
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 python Main.py -data $data -batch $batch -n_head $n_head -n_layers $n_layers -d_model $d_model -d_inner $d_inner