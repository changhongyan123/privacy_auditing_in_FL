from omegaconf import DictConfig, OmegaConf
import hydra
import yaml
import subprocess
from pathlib import Path
import socket
from utilies import config_dir



@hydra.main(version_base=None, config_path="config", config_name="config")
def main(args: DictConfig) -> None:
    """Main function to set up and run federated learning training.
    
    This function:
    1. Updates GPU mapping configuration for distributed training
    2. Loads and modifies training configuration template with provided arguments
    3. Sets up data, model, and training parameters for federated learning
    
    Args:
        args (DictConfig): Hydra configuration object containing all runtime arguments
                          including FL parameters, data settings, and training hyperparameters
    """
    # update the gpu mapping
    num_works = args.fl.num_parties + 1
    config_file_folder = config_dir(args)

    # update the gpu mapping
    with open("config/gpu_mapping.yaml", "r") as f:
        gpu_mapping = yaml.load(f, Loader=yaml.CLoader)

    works_per_gpu = []
    allocated_works = 0
    for _ in range(args.num_gpus - 1):
        works_per_gpu.append(int(num_works / args.num_gpus))
        allocated_works += int(num_works / args.num_gpus)
    works_per_gpu.append(num_works - allocated_works)

    gpu_mapping["mapping_default"][socket.gethostname()] = works_per_gpu

    with open(f"{config_file_folder}/gpu_mapping.yaml", "w") as f:
        print(f"gpu_mapping is saved to {config_file_folder}/gpu_mapping.yaml")
        yaml.dump(gpu_mapping, f, Dumper=yaml.CDumper)

    config_file_path = f"{config_file_folder}/config.yaml"

    with open(args.template_file_name, "r") as f:
        data = yaml.load(f, Loader=yaml.CLoader)

    data["common_args"]["random_seed"] = args.random_seed
    data["save_path"]["server_dir_path"] = args.server_dir_path

    data["data_args"]["dataset"] = args.data.dataset
    data["data_args"]["train_num"] = args.data.train_num
    data["data_args"]["test_num"] = args.data.test_num
    data["data_args"]["pop_num"] = args.data.pop_num
    data["data_args"]["dirichlet_alpha"] = args.fl.dirichlet_alpha
    data["data_args"]["partitioning"] = args.fl.partitioning
    data["data_args"]["train_ratio"] = args.data.train_ratio
    data["data_args"]["pop_ratio"] = args.data.pop_ratio
    
    data["data_args"]["data_argu"] = args.fl.data_argu

    data["model_args"]["model"] = args.fl.model

    data["train_args"]["client_num_in_total"] = args.fl.num_parties
    data["train_args"]["client_num_per_round"] = args.fl.num_parties
    data["train_args"]["comm_round"] = args.fl.comm_round
    data["train_args"]["num_parties"] = args.fl.num_parties
    data["train_args"]["epochs"] = args.fl.epochs
    data["train_args"]["train_batch_size"] = args.fl.train_batch_size
    data["train_args"]["test_batch_size"] = args.fl.test_batch_size
    data["train_args"]["client_optimizer"] = args.fl.client_optimizer
    data["train_args"]["learning_rate"] = args.fl.learning_rate
    data["train_args"]["weight_decay"] = args.fl.weight_decay
    data["train_args"]["batch_size"] = args.fl.batch_size
    data["train_args"]["federated_optimizer"] = args.fl.federated_optimizer

    data["tracking_args"]["wandb_project"] = f"wandb_{args.data.dataset}"
    data["tracking_args"]["save_models"] = args.save_models
    data["tracking_args"]["save_client"] = args.save_client
    data["device_args"]["worker_num"] = args.fl.num_parties + 1
    data["device_args"]["gpu_mapping_file"] = f"{config_file_folder}/gpu_mapping.yaml"


    with open(config_file_path, "w") as f:
        print(f"config is saved to {config_file_path}")
        yaml.dump(data, f, Dumper=yaml.CDumper)

    val = subprocess.check_call(
        f"bash run_mpi.sh {args.fl.num_parties} {config_file_path}", shell=True
    )
    if val == 0:
        print("Job is done")


if __name__ == "__main__":
    main()
