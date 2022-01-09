export CUDA_VISIBLE_DEVICES=3

for run in {1..100}
    do
    python ../tools/al/train_al.py --cfg ../configs/cifar10/al/RESNET18_7percent_IM_START_ONLY.yaml --init random --al dbal --exp-name al_cifar10_random_7percent_im_start_only__deterministic_trial$run
done