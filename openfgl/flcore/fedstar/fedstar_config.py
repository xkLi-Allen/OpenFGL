supported_type_init = ["rw", "dg", "rw_dg"]


config = {
    "n_rw": 16,
    "n_dg": 16, 
    "type_init": "rw_dg"
}


assert config["type_init"] in supported_type_init, "Invalid value of argument 'type_init' for FeStar algorithm."