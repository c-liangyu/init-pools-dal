export CUDA_VISIBLE_DEVICES=2

# for run in {1..100}
for run in {22}
    do
    python ../tools/al/train_al.py --cfg ../configs/cifar10/al/RESNET18_3.5percent_IM_START_ONLY.yaml --init random --al dbal --exp-name al_cifar10_random_3.5percent_im_start_only_trial$run
done