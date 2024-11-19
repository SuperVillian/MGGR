datasets=("BZR"  "DHFR"  "OVCAR-8H" "SF-295H")  
device_id=0
purity=2
degree_purity=True
batch_size=16
lr=0.005  
num_layers=5 
hidden_dim=64 
final_dropout=0.5 

current_time=$(date '+%Y-%m-%d_%H-%M-%S')

for dataset_name in "${datasets[@]}"; do
  log_prefix=${batch_size}_${lr}_${num_layers}_${hidden_dim}_${final_dropout}

  for fold_idx in 0 1 2 3; do
    nohup python -u maingai_addall_feature.py --coarsening_method GBGC --dataset ${dataset_name} --purity ${purity} --degree_purity ${degree_purity} --device ${device_id} --batch_size ${batch_size} --lr ${lr} --num_layers ${num_layers} --hidden_dim ${hidden_dim} --final_dropout ${final_dropout} --fold_idx ${fold_idx} > ./resultstiaocan/${dataset_name}/GBGC/${current_time}_${log_prefix}_${fold_idx}.log 2>&1 &
  done
done


