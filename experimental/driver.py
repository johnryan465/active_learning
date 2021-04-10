from models.model import ModelWrapper
from ignite.engine import Events, Engine
from ignite.metrics import Accuracy, Average
from ignite.handlers import EarlyStopping
from ignite.contrib.handlers import ProgressBar
from ignite.contrib.handlers.tensorboard_logger import global_step_from_engine, TensorboardLogger
from utils.config import IO, VariationalLoss
from models.training import TrainingParams
from ignite.handlers import ModelCheckpoint, Checkpoint

from datasets.activelearningdataset import ActiveLearningDataset
import time
from ray import tune
import torch

class Driver:
    @staticmethod
    def train(exp_name: str, iteration: int, training_params: TrainingParams, model_wrapper: ModelWrapper, dataset: ActiveLearningDataset, tb_logger) -> ModelWrapper:
        training_params = model_wrapper.get_training_params()

        optimizer = model_wrapper.get_optimizer()
        scheduler = model_wrapper.get_scheduler(optimizer)

        train_step = model_wrapper.get_train_step()
        eval_step = model_wrapper.get_eval_step()

        trainer = Engine(train_step)
        evaluator = Engine(eval_step)

        metrics = {
            "training": model_wrapper.get_training_log_hooks(),
            "validation":  model_wrapper.get_test_log_hooks()
        }

        engines = {
            "training": trainer,
            "validation": evaluator
        }

        for stage, engine in engines.items():
            for name, transform in metrics[stage].items():
                if stage == "validation" and (name == "loss" or name == "accuracy"):
                    continue
                metric = Average(output_transform=transform)
                metric.attach(engine, name)

        output_transform = model_wrapper.get_output_transform()  # metrics['validation']['accuracy']

        metric = Accuracy(output_transform=output_transform)
        metric.attach(evaluator, "accuracy")

        loss_fn = model_wrapper.get_loss_fn()
        metric = VariationalLoss(loss_fn)
        metric.attach(evaluator, "loss")

        test_loader = dataset.get_test()
        train_loader = dataset.get_train()

        def score_fn(engine):
            score = engine.state.metrics['accuracy']
            return score

        if training_params.patience > 0:
            es_handler = EarlyStopping(patience=training_params.patience, score_function=score_fn, trainer=trainer)
            evaluator.add_event_handler(Events.COMPLETED, es_handler)

        saving_handler = ModelCheckpoint('/tmp/models', exp_name, n_saved=1, create_dir=True, score_function=score_fn, require_empty=False)
        evaluator.add_event_handler(
            Events.COMPLETED,
            saving_handler,
            to_save = {
                "model": model_wrapper
            }
        )


        for stage, engine in engines.items():
            tb_logger.attach_output_handler(
                engine,
                event_name=Events.EPOCH_COMPLETED,
                tag=stage,
                metric_names=list(metrics[stage].keys()),
                global_step_transform=global_step_from_engine(trainer),
            )

        train_log_lines = []
        test_log_lines = []

        @trainer.on(Events.EPOCH_COMPLETED)
        def log_results(trainer):
            line = dict(trainer.state.metrics)
            line['epoch'] = trainer.state.epoch
            train_log_lines.append(line)

            if scheduler is not None:
                scheduler.step()

        @trainer.on(Events.EPOCH_COMPLETED)
        def eval_results(trainer):
            evaluator.run(test_loader)
            line = dict(evaluator.state.metrics)
            line['epoch'] = trainer.state.epoch
            test_log_lines.append(line)

        if training_params.progress_bar:
            pbar = ProgressBar(dynamic_ncols=True)
            pbar.attach(trainer)
            pbar.attach(evaluator)

        trainer.run(train_loader, max_epochs=training_params.epochs)
        # tune.report(iteration=iteration, mean_loss=test_log_lines[-1]['loss'], accuracy=test_log_lines[-1]['accuracy'])
        model_wrapper.load_state_dict(torch.load(saving_handler.last_checkpoint))
        if training_params.epochs > 0:
            IO.dict_to_csv(train_log_lines, 'experiments/' + exp_name + '/train-' + str(iteration) + '.csv')
            IO.dict_to_csv(test_log_lines, 'experiments/' + exp_name + '/test-' + str(iteration) + '.csv')
        return model_wrapper