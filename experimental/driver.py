from models.model import ModelWrapper
from ignite.engine import Events, Engine
from ignite.metrics import Accuracy, Average
from ignite.handlers import EarlyStopping
from ignite.contrib.handlers import ProgressBar
from ignite.contrib.handlers.tensorboard_logger import global_step_from_engine, TensorboardLogger
from utils.config import IO, VariationalLoss
from models.training import TrainingParams
from ignite.handlers import ModelCheckpoint, Checkpoint
from ignite.metrics.confusion_matrix import ConfusionMatrix
import PIL.Image
from torchvision.transforms import ToTensor
import io

from datasets.activelearningdataset import ActiveLearningDataset
import time
from ray import tune
import matplotlib.pyplot as plt
import numpy as np
import torch
import itertools

class Driver:
    @staticmethod
    def plot_confusion_matrix(cm, class_names):
        """
        Returns a matplotlib figure containing the plotted confusion matrix.
        
        Args:
        cm (array, shape = [n, n]): a confusion matrix of integer classes
        class_names (array, shape = [n]): String names of the integer classes
        """
        
        figure = plt.figure(figsize=(8, 8))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title("Confusion matrix")
        plt.colorbar()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)
        
        # Normalize the confusion matrix.
        cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)
        
        # Use white text if squares are dark; otherwise black.
        threshold = cm.max() / 2.
        
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            color = "white" if cm[i, j] > threshold else "black"
            plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)
            
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        buf = io.BytesIO()
        plt.savefig(buf, format='jpeg')
        buf.seek(0)
        
        image = PIL.Image.open(buf)
        image = ToTensor()(image)
        return image


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
                if stage == "validation" and (name == "loss" or name == "accuracy" or name =="confusion"):
                    continue
                metric = Average(output_transform=transform)
                metric.attach(engine, name)

        output_transform = model_wrapper.get_output_transform()

        metric = Accuracy(output_transform=output_transform)
        metric.attach(evaluator, "accuracy")

        loss_fn = model_wrapper.get_loss_fn()
        metric = VariationalLoss(loss_fn)
        metric.attach(evaluator, "loss")

        metric = ConfusionMatrix(10, output_transform=output_transform)
        metric.attach(evaluator, "confusion")

        test_loader = dataset.get_test()
        train_loader = dataset.get_train()

        if training_params.objective == "accuracy":
            score_fn = lambda engine: engine.state.metrics['accuracy']
        else:
            score_fn = lambda engine: -engine.state.metrics['loss']


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
            print(line)
            line['epoch'] = trainer.state.epoch
            train_log_lines.append(line)

            if scheduler is not None:
                scheduler.step()

        @trainer.on(Events.EPOCH_COMPLETED)
        def eval_results(trainer):
            evaluator.run(test_loader)
            line = dict(evaluator.state.metrics)
            print(line)
            img = Driver.plot_confusion_matrix(line["confusion"].numpy(), [str(i) for i in range(10)])
            tb_logger.writer.add_image("confusion", img, trainer.state.epoch)
            line['epoch'] = trainer.state.epoch
            line['nloss'] = line['loss']
            test_log_lines.append(line)

        if training_params.progress_bar:
            pbar = ProgressBar(dynamic_ncols=True)
            pbar.attach(trainer)
            pbar.attach(evaluator)

        trainer.run(train_loader, max_epochs=training_params.epochs)
        best_epoch = 0
        best_score = float('-inf')
        for i in range(len(test_log_lines)):
            if test_log_lines[i][training_params.objective] > best_score:
                best_epoch = i
                best_score = test_log_lines[i][training_params.objective]
        if training_params.epochs > 0:
            tune.report(iteration=iteration, mean_loss=test_log_lines[best_epoch]['loss'], accuracy=test_log_lines[best_epoch]['accuracy'])
            with torch.no_grad():  
                best_model = model_wrapper.load_state_dict(torch.load(saving_handler.last_checkpoint), dataset)
                del model_wrapper
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            IO.dict_to_csv(train_log_lines, 'experiments/' + exp_name + '/train-' + str(iteration) + '.csv')
            IO.dict_to_csv(test_log_lines, 'experiments/' + exp_name + '/test-' + str(iteration) + '.csv')
            return best_model
        else:
            return model_wrapper
