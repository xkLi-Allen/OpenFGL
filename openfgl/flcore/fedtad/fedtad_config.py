
supported_distill_mode = ["rep_distill", "raw_distill"]

config = {
    "save_ckr": False,
    "max_walk_length": 5,
    "distill_mode": "raw_distill",
    "topk": 5,
    "num_gen": 100,
    "noise_dim": 32,
    "gen_dropout": 0,
    "glb_epochs": 5,
    "it_g": 1,
    "it_d": 5,
    "lr_g": 1e-3,
    "lr_d": 1e-3,
    "lam1": 1,
    "lam2": 1  
}


assert config["distill_mode"] in supported_distill_mode