tot_pop=10
tot_seed=2
pop_size=1
while [ $pop_size -le $tot_pop ]
do
 seed=0
 while [ $seed -le $tot_seed ]
 do
   python -m egg.zoo.population_signal_game.train --root=data_prep/my_imagenet_features --game_size 3 --vocab_size 5 --n_epochs 300 --lr 1e-3 --max_len 5 --seed $seed --pop_mode 0 --pop_size $pop_size --multi_head 0 --exp_prefix noadv-task-1
  ((seed++))
 done
 ((pop_size++))
done


#
#
# pop_mode=0
# while [ $pop_mode -le 2 ]
# do
#  seed=0
#  while [ $seed -le 3 ]
#  do
#    python -m egg.zoo.population_signal_game.train --root=data_prep/my_imagenet_features --game_size 3 --vocab_size 5 --n_epochs 200 --lr 1e-3 --max_len 5 --seed $seed --pop_mode $pop_mode --pop_size 3
#   ((seed++))
#  done
#  ((pop_mode++))
# done
#
# echo alldone
