import pytest
import os
from shutil import rmtree
from phd_lab.experiments.trainer import Trainer
from phd_lab.metrics import Accuracy, Top5Accuracy
from phd_lab.models import vgg11
from phd_lab.datasets.cifar import Cifar10
from phd_lab.optimizers import adam
import torch


@pytest.fixture()
def trainer():
    model = vgg11(num_classes=10)
    return Trainer(
        model=model,
        data_bundle=Cifar10(batch_size=128, output_size=32, cache_dir='tmp'),
        optimizer_bundle=adam(model=model),
        run_id="test",
        batch_size=128,
        epochs=1,
        metrics=[Accuracy(), Top5Accuracy()],
        logs_dir='./test_logs/',
        device='cuda:0' if torch.cuda.is_available() else 'cpu'
    )


class TestInit:

    def test_folder_structure_correct(self, trainer):
        assert os.path.exists('./test_logs/')
        rmtree('./test_logs')

    def test_folder_structure_deep_correct(self, trainer):
        assert os.path.exists('./test_logs/VGG11/Cifar10_32/test')
        rmtree('./test_logs/')


class TestTrain:

    def test_training(self, trainer):
        trainer.train()
        assert os.path.exists(trainer._save_path)

    def test_resume_training(self, trainer):
        trainer.train()
        assert os.path.exists(trainer._save_path)
