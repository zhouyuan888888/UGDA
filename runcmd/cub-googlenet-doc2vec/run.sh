PROJECT="cub-googlenet-doc2vec"
NOTES="fpml"
MiSSRATES=(0 0.1 0.2 0.3 0.4 0.5)

for eta in ${MiSSRATES[*]}
do
python3 -m scripts.cub_googlenet_doc2vec.main \
    -F logs/ugda/cub_googlenet_doc2vec \
    with dataset.path="data/cub_googlenet_doc2vec" \
    dataset.batch_size=1 \
    gpu_id=0 \
    gpu_shift=0 \
    seed=2048 \
    Project=${PROJECT} \
    Notes=${NOTES} \
    eval.missing_rate=${eta} \
    ugda.iter=1000 \
    ugda.rec_iter=30 \
    eval.task_num=400 \
    eval.batch_num=400 \
    eval.sampled_num=100 \
    eval.way=3 \
    eval.shot=1 \
    eval.query_num=15 \
    eval.latent_dim=1024 \
    eval.beta=1e-3 \
    eval.topk=1 \
    ugda.lr=1e-2 \
    ugda.rec_lr=1e-2 \
    eval.refresh_task=True \
    eval.refresh_gaussian=True \
    eval.tukey_trans=1 \
    dis_dir="ckpt/cub_googlenet_doc2vec/dis/" \
    gaussian_dir="ckpt/cub_googlenet_doc2vec/Gaussian/" \
    test_dir="data/cub_googlenet_doc2vec/test/" \
    task_dir="data/cub_googlenet_doc2vec/task/" \
    base_dir="data/cub_googlenet_doc2vec/base/"
done