pop_mode=0
while [ $pop_mode -le 2 ]
do
 seed=0
 while [ $seed -le 3 ]
 do
   !python -m egg.zoo.population_signal_game.train --root=data_prep/my_imagenet_features --game_size 3 --vocab_size 5 --n_epochs 200 --lr 1e-3 --max_len 5 --seed $seed --pop_mode $pop_mode --pop_size 1
  ((seed++))
 done
 ((dataset_type++))
done




pop_mode=0
while [ $pop_mode -le 2 ]
do
 seed=0
 while [ $seed -le 3 ]
 do
   !python -m egg.zoo.population_signal_game.train --root=data_prep/my_imagenet_features --game_size 3 --vocab_size 5 --n_epochs 200 --lr 1e-3 --max_len 5 --seed $seed --pop_mode $pop_mode --pop_size 3
  ((seed++))
 done
 ((dataset_type++))
done

echo alldone
