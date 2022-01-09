export CUDA_VISIBLE_DEVICES=2
for run in {1..100}
# for run in {51..100}
    do
    python ../tools/al/train_al.py --cfg ../configs/cifar10/al/RESNET18_0.2percent_START_ONLY.yaml --init random --al dbal --exp-name al_cifar10_random_0.2percent_start_only_trial$run
done