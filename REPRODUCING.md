## Create model
python src/mnist_mlp_train.py --seed 0 --optimizer sgd --learning-rate 1e-3 --test

## Weight matching
python src/mnist_mlp_weight_matching.py --model-a mlp_0 --model-b mlp_1