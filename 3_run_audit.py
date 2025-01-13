import os

import numpy as np
import torch
import hydra
from omegaconf import DictConfig
import numpy as np
from utilies import (
    compute_p_value,
    get_low,
    get_trend_slope,
    config_dir,
    population_attack,
    get_list_dict,
    get_dataset_classes,
)
from utilies import save_info_results_combined
import jsonlines


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(args: DictConfig) -> None:

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.audit.gpu_id}")
    else:
        device = torch.device("cpu")

    num_classes = get_dataset_classes(args)

    log_dir = config_dir(args)
    if args.audit.party == "all":
        clients_list = [i for i in range(args.fl.num_parties)]
    else:
        clients_list = [args.audit.party]

    if args.audit.alg == "all":
        all_alg = [
            "fed-loss",  # average loss over the whole sequence
            "fed-conf",  # average confidence over the whole sequence
            "fed-rescaled",  # average rescaled over the whole sequence
            "delta-diff",  # delta-diff from (How to Combine Membership-Inference Attacks on Multiple Updated Models(https://arxiv.org/abs/2205.06369))
            "delta-ratio",  # delta-ratio from
            "back-front-diff",  # back-front-diff from (How to Combine Membership-Inference Attacks on Multiple Updated Models(https://arxiv.org/abs/2205.06369))
            "back-front-ratio",  # back-front-ratio from (How to Combine Membership-Inference Attacks on Multiple Updated Models(https://arxiv.org/abs/2205.06369))
            "01-loss",  # loss in the last round
            "our_loss",  # trend slope of loss
            "our_conf",  # trend slope of confidence
            "our_rescaled",  # trend slope of rescaled
        ]

    else:
        raise ValueError(f"alg {args.audit.alg} not supported")

    for random_seed in args.audit.random_seed_list:
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        list_dict = get_list_dict(args, random_seed)
        # training reference models for each client:

        for c_idx in clients_list:
            all_loss = []
            all_conf = []
            all_rescaled = []
            all_acc = []
            file_path = f"{log_dir}/{c_idx}_{args.audit.target_model_type}_pred.jsonl"
            if os.path.exists(file_path):
                with jsonlines.open(file_path) as reader:
                    for obj in reader:
                        # Process each object here
                        t = int(list(obj.keys())[0])
                        if args.audit.start_round > t:
                            continue
                        elif args.audit.target_round < t:
                            break
                        all_loss.append(obj[str(t)]["loss"])
                        all_conf.append(obj[str(t)]["conf"])
                        all_rescaled.append(obj[str(t)]["rescaled"])
                        all_acc.append(obj[str(t)]["acc"])

                all_loss = np.array(all_loss).T
                all_conf = -np.array(all_conf).T
                all_rescaled = -np.array(all_rescaled).T
                all_acc = np.array(all_acc).T
                membership = np.concatenate(
                    [
                        len(list_dict[c_idx]["train"]) * [1],
                        len(list_dict[c_idx]["test"]) * [0],
                        len(list_dict[c_idx]["rest"]) * [-1],
                    ]
                )

                # different alg
                for alg in all_alg:
                    print(f"Auditing privacy risk using {alg} for client {c_idx}")
                    if alg == "01-acc":
                        target_signal = all_acc[:, -1]
                    if alg == "01-loss":
                        target_signal = all_loss[:, -1]
                    elif alg == "back-front-diff":
                        loss = all_loss[:, -1]
                        loss_init = all_loss[:, 0]
                        loss_diff = loss - loss_init
                        target_signal = loss_diff

                    elif alg == "back-front-ratio":
                        loss_init = all_loss[:, 0]
                        loss = all_loss[:, -1]
                        target_signal = (loss + 1e-45) / (loss_init + 1e-45)
                    # multiple case
                    elif alg == "fed-loss":
                        target_signal = np.mean(all_loss, axis=1)
                    elif alg == "fed-conf":
                        target_signal = np.mean(all_conf, axis=1)
                    elif alg == "fed-rescaled":
                        target_signal = np.mean(all_rescaled, axis=1)
                    elif "delta" in alg:
                        # f1-f0
                        if alg == "delta-diff":
                            loss_diff = all_loss[:, 1:] - all_loss[:, :-1]
                        elif alg == "delta-ratio":
                            loss_diff = (all_loss[:, 1:] + 1e-45) / (
                                all_loss[:, :-1] + 1e-45
                            )

                        target_diff = loss_diff[membership != -1]
                        target_membership = membership[membership != -1]
                        pop_diff = loss_diff[membership == -1]
                        target_p = np.array(
                            [
                                compute_p_value(target_diff[:, t], pop_diff[:, t])
                                for t in range(
                                    args.audit.target_round - args.audit.start_round
                                )
                            ]
                        ).T

                        pop_p = np.array(
                            [
                                compute_p_value(pop_diff[:, t], pop_diff[:, t])
                                for t in range(
                                    args.audit.target_round - args.audit.start_round
                                )
                            ]
                        ).T

                        target_min_p = np.min(target_p, axis=1)
                        pop_min_p = np.min(pop_p, axis=1)
                        target_signal = np.concatenate(
                            [target_min_p, pop_min_p], axis=0
                        )
                        assert (
                            np.mean(
                                np.concatenate(
                                    [target_membership, np.ones(len(pop_diff)) * -1],
                                    axis=0,
                                )
                                - membership
                            )
                            == 0
                        )

                    elif alg == "our_loss":
                        target_signal = get_trend_slope(all_loss)
                    elif alg == "our_conf":
                        target_signal = get_trend_slope(all_conf)
                    elif alg == "our_rescaled":
                        target_signal = get_trend_slope(all_rescaled)
                    else:
                        raise ValueError(f"alg {alg} not supported")

                    train_acc = all_acc[membership == 1].mean()
                    test_acc = all_acc[membership == 0].mean()
                    train_loss = all_loss[membership == 1].mean()
                    test_loss = all_loss[membership == 0].mean()
                    tpr, fpr, auc_pop = population_attack(
                        target_signal[membership == 1],
                        target_signal[membership == 0],
                        target_signal[membership == -1],
                    )

                    save_info_results_combined(
                        args,
                        alg,
                        random_seed,
                        c_idx,
                        auc_pop,
                        train_acc,
                        test_acc,
                        train_loss,
                        test_loss,
                        get_low(tpr, fpr, 0.001)[0],
                        get_low(tpr, fpr, 0.005)[0],
                        get_low(tpr, fpr, 0.01)[0],
                        get_low(tpr, fpr, 0.02)[0],
                        get_low(tpr, fpr, 0.05)[0],
                    )
            else:
                raise ValueError(f"file not found: {file_path}")


if __name__ == "__main__":
    main()
