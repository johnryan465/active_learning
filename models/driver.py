from models.model import ModelWrapper
from ignite.engine import Events, Engine
from ignite.metrics import Accuracy, Average, Loss
from ignite.contrib.handlers import ProgressBar
from ignite.contrib.handlers.tensorboard_logger import *

from datasets.activelearningdataset import ActiveLearningDataset
import time

class Driver:
    @staticmethod
    def train(model_wrapper: ModelWrapper, dataset: ActiveLearningDataset):
        training_params = model_wrapper.get_training_params()

        optimizer = model_wrapper.get_optimizer()
        scheduler = model_wrapper.get_scheduler(optimizer)
        train_step = model_wrapper.get_train_step()
        eval_step = model_wrapper.get_eval_step()

        trainer = Engine(train_step)
        evaluator = Engine(eval_step)
        def output_transform_loss(output):
            return output['loss']
        def output_transform_cond(output):
            return output['cond']
        def output_transform_min(output):
            return output['min']
        metric = Average(output_transform=output_transform_loss)
        metric.attach(trainer, "loss")

        metric = Average(output_transform=output_transform_cond)
        metric.attach(trainer, "cond")

        metric = Average(output_transform=output_transform_min)
        metric.attach(trainer, "min")

        output_transform = model_wrapper.get_output_transform()

        metric = Accuracy(output_transform=output_transform)
        metric.attach(evaluator, "accuracy")

        loss_fn = model_wrapper.get_loss_fn()
        metric = Loss(loss_fn)
        metric.attach(evaluator, "loss")

        test_loader = dataset.get_test()
        train_loader = dataset.get_train()

        ts = time.time()
        tb_logger = TensorboardLogger(log_dir="logs/" +str(ts))

        for tag, evaluator in [("training", trainer), ("validation", evaluator)]:
            tb_logger.attach_output_handler(
                evaluator,
                event_name=Events.EPOCH_COMPLETED,
                tag=tag,
                metric_names=["loss", "accuracy", "cond", "min"],
                global_step_transform=global_step_from_engine(trainer),
            )

        @trainer.on(Events.EPOCH_COMPLETED)
        def log_results(trainer):
            evaluator.run(test_loader)
            if scheduler is not None:
                scheduler.step()

        pbar = ProgressBar(dynamic_ncols=True)
        pbar.attach(trainer)

        trainer.run(train_loader,
                   max_epochs=training_params.epochs)

        tb_logger.close()


    @staticmethod
    def test(model_wrapper: ModelWrapper, dataset: ActiveLearningDataset):
        training_params = model_wrapper.get_training_params()

        optimizer = model_wrapper.get_optimizer()
        scheduler = model_wrapper.get_scheduler(optimizer)
        train_step = model_wrapper.get_train_step(optimizer)
        eval_step = model_wrapper.get_eval_step()

        trainer = Engine(train_step)
        evaluator = Engine(eval_step)

        metric = Average()
        metric.attach(trainer, "loss")

        output_transform = model_wrapper.get_output_transform()

        metric = Accuracy(output_transform=output_transform)
        metric.attach(evaluator, "accuracy")

        loss_fn = model_wrapper.get_loss_fn()
        metric = Loss(loss_fn)
        metric.attach(evaluator, "loss")

        test_loader = dataset.get_test()
        train_loader = dataset.get_train()

        @trainer.on(Events.EPOCH_COMPLETED)
        def log_results(trainer):
            metrics = trainer.state.metrics
            print("Train", metrics)

            evaluator.run(test_loader)
            metrics = evaluator.state.metrics

            print("Test", metrics)

            if scheduler is not None:
                scheduler.step()

        pbar = ProgressBar(dynamic_ncols=True)
        pbar.attach(trainer)

        # trainer.run(train_loader,
        #            max_epochs=training_params.epochs)
