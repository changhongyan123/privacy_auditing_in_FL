import fedml
from fedml import FedMLRunner
from fedml.model.cv.resnet_gn import resnet18
from data_loader.data_loader import load_data
from utilies import get_save_dir
from model import FullyConnectedModel, get_model

if __name__ == "__main__":
    # init FedML framework
    args = fedml.init()
    args.save_dir = get_save_dir(args, is_train=True)
    # init device
    device = fedml.device.get_device(args)

    # load data
    dataset, output_dim = load_data(args)

    # load model
    if "resnet" in args.model:
        model = fedml.model.create(args, output_dim)
    elif "purchase" in args.dataset and args.model == "nn":
        model = FullyConnectedModel(
            input_size=600, hidden_sizes=[1024, 512, 256, 128], output_size=100
        )
    elif "texas"  in args.dataset and args.model == "nn":
        model = FullyConnectedModel(
            input_size=6169, hidden_sizes=[1024, 512, 256, 128], output_size=100
        )
    elif args.dataset == "texas" and args.model == "nn3":
        model = FullyConnectedModel(
            input_size=6169, hidden_sizes=[512, 256, 128], output_size=100
        )

    elif args.dataset == "texas" and args.model == "nn2":
        model = FullyConnectedModel(
            input_size=6169, hidden_sizes=[256, 128], output_size=100
        )

    elif args.dataset == "texas" and args.model == "nn1":
        model = FullyConnectedModel(
            input_size=6169, hidden_sizes=[128], output_size=100
        )
    else:
        model = get_model(args.model, output_dim, args.dataset)

    fedml_runner = FedMLRunner(args, device, dataset, model)
    fedml_runner.run()
