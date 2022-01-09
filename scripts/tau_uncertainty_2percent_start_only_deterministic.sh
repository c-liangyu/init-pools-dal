export CUDA_VISIBLE_DEVICES=3
# for run in {1..50}
for run in {1..10}
    do
    python ../tools/al/train_al.py --cfg ../configs/cifar10/al/RESNET18_2percent_START_ONLY.yaml --init simclr --simclr-duplicate /media/ntu/volume2/home/s121md302_06/workspace/code/temperature-as-uncertainty-public/experiments/tau_simclr_base/viz/tau_uncertainty.npy --al dbal --exp-name al_cifar10_tau_uncertainty_2percent_start_only__deterministic_trial$run
done