PROJECT="ORL"
NOTES="fpml"
MissRates=(0 0.1 0.2 0.3 0.4 0.5)

for eta in ${MissRates[*]}
do
python3 -m scripts.orl.main \
    with dataset.path="data/orl" \
    dataset.batch_size=1 \
    gpu_id=0 \
    gpu_shift=0 \
    seed=2048 \
    Project=${PROJECT} \
    Notes=${NOTES} \
    ugda.iter=1000 \
    ugda.rec_iter=30 \
    ugda.temp=1 \
    ugda.update_net_iter=10 \
    ugda.update_latent_iter=10 \
    ugda.alpha=1 \
    eval.task_num=400 \
    eval.batch_num=50 \
    eval.sampled_num=50 \
    eval.way=5 \
    eval.shot=1 \
    eval.query_num=9 \
    eval.missing_rate=${eta} \
    eval.latent_dim=1024 \
    eval.beta=1e-3 \
    eval.topk=1 \
    ugda.lr=1e-2 \
    ugda.rec_lr=1e-2 \
    eval.refresh_task=True \
    eval.refresh_gaussian=True \
    eval.tukey_trans=1 \
    dis_dir="ckpt/orl/dis/" \
    gaussian_dir="ckpt/orl/Gaussian/" \
    test_dir="data/orl/test/" \
    task_dir="data/orl/task/" \
    base_dir="data/orl/base/"
 done
