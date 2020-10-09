tot_pop=10
tot_seed=3
pop_size=1
while [ $pop_size -le $tot_pop ]
do
 seed=0
 while [ $seed -le $tot_seed ]
 do
   python -m egg.zoo.population_signal_game.train --root=data_prep/my_imagenet_features --game_size 3 --vocab_size 5 --n_epochs 200 --lr 1e-3 --max_len 5 --seed $seed --pop_mode 0 --pop_size $pop_size --multi_head 2 --exp_prefix adv-task-1
  ((seed++))
 done
 ((pop_mode++))
done
