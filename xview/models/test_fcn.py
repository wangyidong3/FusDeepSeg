from .fcn import FCN
from xview.datasets.synthia import Synthia


def test_can_build_model():
    net = FCN(num_classes=1, num_units=64)
    net.close()
    assert True


def test_can_export_and_import_weights():
    with FCN(num_classes=1, num_units=64) as net:
        path = net.export_weights(save_dir='/tmp/')
        net.import_weights(path)
        assert True


def test_can_run_training():
    data = Synthia(['UNITTEST-SEQUENCE'], 2)
    config = {'num_classes': 14,
              'num_units': 5,
              'dropout_rate': 0.2,
              'learning_rate': 0.01}
    with FCN(**config) as net:
        net.fit(data, 1)
