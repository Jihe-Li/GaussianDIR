num_pair=10

for i in $(seq 1 $num_pair)
do
python train.py dataset.case_idx=$i 

done
