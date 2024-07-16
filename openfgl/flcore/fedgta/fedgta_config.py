supported_moment_type = ["raw", "central", "hybrid"]


config = {
    "prop_steps": 5, 
    "lp_alpha": 0.5, 
    "temperature": 20, 
    "num_moments": 10, 
    "moment_type": "hybrid",
    "accept_alpha": 0.5
}

assert config["moment_type"] in supported_moment_type, "Invalid value of argument 'moment_type' for FedGTA algorithm."