from .deeplab import Deeplab
from xview.datasets.synthia import Synthia

config = {'num_classes': 14,
          'num_units': 5,
          'dropout_rate': 0.2,
          'learning_rate': 0.01,
          'num_channels': 3,
          'modality': 'rgb',
          'batch_normalization' : True}


def test_can_build_model():
    net = Deeplab(**config)
    net.close()
    assert True


def test_can_run_training():
    data = Synthia(['UNITTEST-SEQUENCE'], 2)

    with Deeplab(**config) as net:
        net.fit(data, 1)
