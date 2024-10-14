dataset_name="DD"
device_id=1
gc_method='GBGC'
degree_purity=True
# Purity usage: 0 is not used, 2 is purity adaptive, 3 is purity plus structure adaptive, 0-1 is purity setting
purity=2 

nohup python -u main.py --dataset ${dataset_name} --coarsening_method ${gc_method} --purity ${purity} --degree_purity ${degree_purity} --device ${device_id} --fold_idx 0 > ./result/${dataset_name}_0.log 2>&1 &
nohup python -u main.py --dataset ${dataset_name} --coarsening_method ${gc_method} --purity ${purity} --degree_purity ${degree_purity} --device ${device_id} --fold_idx 1 > ./result/${dataset_name}_1.log 2>&1 &
nohup python -u main.py --dataset ${dataset_name} --coarsening_method ${gc_method} --purity ${purity} --degree_purity ${degree_purity} --device ${device_id} --fold_idx 2 > ./result/${dataset_name}_2.log 2>&1 &
nohup python -u main.py --dataset ${dataset_name} --coarsening_method ${gc_method} --purity ${purity} --degree_purity ${degree_purity} --device ${device_id} --fold_idx 3 > ./result/${dataset_name}_3.log 2>&1 &
