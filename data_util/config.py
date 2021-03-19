import os
from collections import namedtuple

root_dir = os.path.expanduser("~")

#train_data_path = os.path.join(root_dir, "ptr_nw/cnn-dailymail-master/finished_files/train.bin")
train_data_path = os.path.join(root_dir, "Code/ML/courses/3_semester/Seminar/dataset/finished_files/test.bin")
eval_data_path = os.path.join(root_dir, "Code/ML/courses/3_semester/Seminar/dataset/finished_files/val.bin")
decode_data_path = os.path.join(root_dir, "Code/ML/courses/3_semester/Seminar/dataset/finished_files/test.bin")
vocab_path = os.path.join(root_dir, "Code/ML/courses/3_semester/Seminar/dataset/finished_files/vocab")
log_root = os.path.join(root_dir, "/home/denys/Code/ML/courses/3_semester/Seminar/seminar/log")

# Hyperparameters
hidden_dim=256
emb_dim=128
batch_size=4
max_enc_steps=20
max_dec_steps=20
padding=max_enc_steps
beam_size=4
min_dec_steps=5
vocab_size=5000

#Optimizer
lr=0.15
adagrad_init_acc=0.1
rand_unif_init_mag=0.02
trunc_norm_init_std=1e-4
max_grad_norm=2.0

pointer_gen = False
is_coverage = False
cov_loss_wt = 1.0

eps = 1e-12
max_iterations = 5
use_gpu=True
lr_coverage=0.15

hps_dict = {"mode": "train",
            "hidden_dim": hidden_dim,
            "emb_dim": emb_dim,
            "batch_size": batch_size,
            "max_enc_steps": max_enc_steps,
            "max_dec_steps": max_dec_steps,
            "beam_size": beam_size,
            "min_dec_steps": min_dec_steps,
            "vocab_size": vocab_size,
            "lr": lr,
            "adagrad_init_acc": adagrad_init_acc,
            "rand_unif_init_mag": rand_unif_init_mag,
            "trunc_norm_init_std": trunc_norm_init_std,
            "max_grad_norm": max_grad_norm,
            "pointer_gen": pointer_gen,
            "is_coverage": is_coverage,
            "cov_loss_wt": cov_loss_wt}

hps = namedtuple("HParams", hps_dict.keys())(**hps_dict)