import torch
from torch import nn

from ...core.alg_frame.client_trainer import ClientTrainer
from ...core.dp.fedml_differential_privacy import FedMLDifferentialPrivacy
import logging
import copy
import logging
import time
import numpy as np

# This is modified.
from scipy.special import softmax


class ModelTrainerCLS(ClientTrainer):
    def get_model_params(self):
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters)

    def train(self, train_data, device, args):
        model = self.model

        model.to(device)
        model.train()

        # train and update
        criterion = nn.CrossEntropyLoss().to(device)  # pylint: disable=E1102
        if "sgd" in args.client_optimizer:
            optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=args.learning_rate,
                weight_decay=args.weight_decay,
            )
        else:
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=args.learning_rate,
                weight_decay=args.weight_decay,
                amsgrad=True,
            )

        epoch_loss = []
        for epoch in range(args.epochs):
            batch_loss = []

            for batch_idx, (x, labels) in enumerate(train_data):
                x, labels = x.to(device), labels.to(device)
                # model.zero_grad()
                log_probs = model(x)
                labels = labels.long()
                loss = criterion(log_probs, labels)  # pylint: disable=E1102

                loss.backward()

                if "fed" not in args.client_optimizer:
                    # logging.info("Optimizer step")
                    optimizer.step()
                    optimizer.zero_grad()

            if "fed" in args.client_optimizer:
                logging.info("federated steps")
                optimizer.step()
                optimizer.zero_grad()
                batch_loss.append(loss.item())
            if len(batch_loss) == 0:
                epoch_loss.append(0.0)
            else:
                epoch_loss.append(sum(batch_loss) / len(batch_loss))
            logging.info(
                "Client Index = {}\tEpoch: {}\tLoss: {:.6f}".format(
                    self.id, epoch, sum(epoch_loss) / len(epoch_loss)
                )
            )

    def train_iterations(self, train_data, device, args):
        model = self.model

        model.to(device)
        model.train()

        # train and update
        criterion = nn.CrossEntropyLoss().to(device)  # pylint: disable=E1102
        if "sgd" in args.client_optimizer:
            optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=args.learning_rate,
                momentum=0.9,
                weight_decay=args.weight_decay,
            )
        else:
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=args.learning_rate,
                weight_decay=args.weight_decay,
                amsgrad=True,
            )

        epoch_loss = []

        current_steps = 0
        current_epoch = 0
        while current_steps < args.local_iterations:
            batch_loss = []
            for batch_idx, (x, labels) in enumerate(train_data):
                x, labels = x.to(device), labels.to(device)

                log_probs = model(x)
                labels = labels.long()
                loss = criterion(log_probs, labels)  # pylint: disable=E1102
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                batch_loss.append(loss.item())
                current_steps += 1
                if current_steps == args.local_iterations:
                    break
            current_epoch += 1
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            logging.info(
                "Client Index = {}\tEpoch: {}\tLoss: {:.6f}".format(
                    self.id, current_epoch, sum(epoch_loss) / len(epoch_loss)
                )
            )

    def test(self, test_data, device, args, is_test=False):
        ## Modified: save the per sample loss in each round.
        model = self.model
        start_time = time.time()
        model.to(device)
        model.eval()
        if is_test:
            rescaled_list = []
            logits_list = []
            loss_list = []
            conf_list = []
            acc_list = []

        metrics = {"test_correct": 0, "test_loss": 0, "test_total": 0}
        if is_test:
            criterion = nn.CrossEntropyLoss(reduction="none").to(device)
        else:
            criterion = nn.CrossEntropyLoss().to(device)
        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(test_data):
                COUNT = len(x)
                x = x.to(device)
                target = target.to(device)
                pred = model(x)
                target = target.long()
                loss = criterion(pred, target)  # pylint: disable=E1102

                _, predicted = torch.max(pred, -1)
                correct = predicted.eq(target)
                metrics["test_correct"] += correct.sum().item()
                metrics["test_loss"] += loss.sum().item()
                metrics["test_total"] += target.size(0)

                if is_test:
                    loss_list.append(loss)
                    acc_list.append(correct)
                    target = target.to("cpu")
                    logits_list.append(pred[np.arange(COUNT), target])
                    confi = softmax(pred.detach().cpu().numpy(), axis=1)
                    confi_corret = confi[np.arange(COUNT), target]
                    conf_list.append(confi_corret)
                    confi[np.arange(COUNT), target] = 0
                    confi_wrong = np.sum(confi, axis=1)
                    logit = np.log(confi_corret + 1e-45) - np.log(confi_wrong + 1e-45)
                    rescaled_list.append(logit)

        if is_test:
            loss_list = torch.cat(loss_list, dim=0).detach().cpu().numpy()
            logits_list = torch.cat(logits_list, dim=0).detach().cpu().numpy()
            conf_list = np.concatenate(conf_list)
            rescaled_list = np.concatenate(rescaled_list)
            acc_list = torch.cat(acc_list, dim=0).detach().cpu().numpy()
            metrics["save_information"] = {
                "loss": loss_list.tolist(),
                "logits": logits_list.tolist(),
                "conf": conf_list.tolist(),
                "rescaled": rescaled_list.tolist(),
                "acc": acc_list.tolist(),
            }
            logging.info(f"time for testing test loader: {time.time() - start_time}")
        else:
            logging.info(
                "time for training loader: {}".format(time.time() - start_time)
            )
        return metrics
