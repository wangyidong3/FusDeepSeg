import numpy as np
from os import listdir, path, makedirs, environ
from tqdm import tqdm
import tarfile
import cv2
import shutil
import json
from sklearn.model_selection import train_test_split
from copy import deepcopy

from .data_baseclass import DataBaseclass
from .augmentation import augmentate
from .synthia import SYNTHIA_BASEPATH, \
    one_channel_image_reader


# Set label information according to synthia README
LABELINFO = {
   0: {'name': 'void', 'color': [0, 0, 0]},
   1: {'name': 'sky', 'color': [128, 128, 128]},
   2: {'name': 'building', 'color': [128, 0, 0]},
   3: {'name': 'road', 'color': [128, 64, 128]},
   4: {'name': 'sidewalk', 'color': [0, 0, 192]},
   5: {'name': 'fence', 'color': [64, 64, 128]},
   6: {'name': 'vegetation', 'color': [128, 128, 0]},
   7: {'name': 'pole', 'color': [192, 192, 128]},
   8: {'name': 'car', 'color': [64, 0, 128]},
   9: {'name': 'traffic sign', 'color': [192, 128, 128]},
   10: {'name': 'pedestrian', 'color': [64, 64, 0]},
   11: {'name': 'bicycle', 'color': [0, 128, 192]}
}


class SynthiaCityscapes(DataBaseclass):
    """Driver for SYNTHIA dataset (http://synthia-dataset.net/)."""

    _data_shape_description = {
            'rgb': (None, None, 3), 'depth': (None, None, 1), 'labels': (None, None)}
    _num_default_classes = 12

    def __init__(self, base_path=SYNTHIA_BASEPATH, force_preprocessing=False,
                 batchsize=1, resize=False, in_memory=False, **data_config):

        config = {
            'augmentation': {
                'crop': [1, 240],
                'scale': [.4, 0.7, 1.5],
                'vflip': .3,
                'hflip': False,
                'gamma': [.4, 0.3, 1.2],
                'rotate': [.4, -13, 13],
                'shear': [0, 0.01, 0.03],
                'contrast': [.3, 0.5, 1.5],
                'brightness': [.2, -40, 40]
            },
            'labels': {
                'lanemarkings': False
            }
        }
        config.update(data_config)
        config.update({'resize': resize})
        self.config = config

        if not path.exists(base_path):
            message = 'ERROR: Path to SYNTHIA dataset does not exist.'
            print(message)
            raise IOError(1, message, base_path)

        self.basepath = path.join(base_path, 'RAND_CITYSCAPES')

        # Every sequence got their own train/test split during preprocessing. According
        # to the loaded sequences, we now collect all files from all sequence-subsets
        # into one list.
        if in_memory and 'TMPDIR' in environ:
            print('INFO loading dataset into memory')
            # first load the tarfile into a closer memory location, then load all the
            # images
            tar = tarfile.open(path.join(SYNTHIA_BASEPATH, 'RAND_CITYSCAPES.tar.gz'))
            localtmp = environ['TMPDIR']
            tar.extractall(path=localtmp)
            tar.close()
            self.basepath = localtmp
            with open(path.join(self.basepath, 'train_test_split.json'), 'r') as f:
                split = json.load(f)
                trainset = [{'image': self._load_data(filename)}
                            for filename in tqdm(split['trainset'])]
                testset = [{'image': self._load_data(filename)}
                           for filename in tqdm(split['testset'])]
        else:
            if in_memory:
                print('INFO Environment Variable TMPDIR not set, could not unpack data '
                      'and load into memory\n'
                      'Now trying to load every image seperately')
            with open(path.join(self.basepath, 'train_test_split.json'), 'r') as f:
                split = json.load(f)
                trainset = [{'image_name': filename} for filename in split['trainset']]
                testset = [{'image_name': filename} for filename in split['testset']]

        measureset, testset = train_test_split(testset, test_size=0.5, random_state=1)

        # Update labelinfo according to config
        labelinfo = deepcopy(LABELINFO)
        if self.config['labels']['lanemarkings']:
            labelinfo[12] = {'name': 'lanemarking', 'color': [0, 192, 0]}

        # Intitialize Baseclass
        DataBaseclass.__init__(self, trainset, measureset, testset, labelinfo)

    @property
    def one_hot_lookup(self):
        return np.arange(len(self.labelinfo), dtype=np.int)

    def _preprocessing(self, sequence):
        rootpath = path.join(self.basepath, sequence, 'GT')

        for direction in ['F', 'B', 'L', 'R']:
            inpath, outpath = (path.join(rootpath, pref,
                                         'Stereo_Right/Omni_{}'.format(direction))
                               for pref in ['LABELS', 'LABELS_NPY'])

            if path.exists(outpath):
                shutil.rmtree(outpath)
            makedirs(outpath)
            for filename in tqdm(listdir(inpath)):
                array = one_channel_image_reader(path.join(inpath, filename),
                                                 np.uint8)
                np.save(path.join(outpath, filename.split('.')[0]), array)

            if sequence == 'RAND_CITYSCAPES':
                # There are no different directions for this sequence.
                break

        # create train-test-split if necessary
        if not path.exists(path.join(self.basepath, sequence, 'train_test_split.json')):
            print("INFO: Creating Train-Test-Split")
            filenames = [filename.split('.')[0] for filename
                         in listdir(path.join(rootpath, 'LABELS/Stereo_Right/Omni_F'))]
            trainset, testset = train_test_split(filenames, test_size=0.2)
            with open(path.join(self.basepath, sequence, '/train_test_split.json'),
                      'w') as f:
                json.dump({'trainset': trainset, 'testset': testset}, f)

    def _load_data(self, image_name):
        filetype = {'rgb': 'png', 'depth': 'png', 'labels': 'npy'}

        rgb_filename, depth_filename, groundtruth_filename = (
            path.join(self.basepath, '{}/Stereo_Right/Omni_F/{}.{}'
                      .format(pref, image_name, filetype[modality]))
            for pref, modality in zip(['RGB', 'Depth', 'GT/LABELS_NPY'],
                                      ['rgb', 'depth', 'labels']))

        blob = {}
        blob['rgb'] = cv2.imread(rgb_filename)
        # flag 2 -> read image with 16bit depth
        blob['depth'] = cv2.imread(depth_filename, 2)
        labels = np.load(groundtruth_filename)
        # Dirty fix for the class mappings as in adapnet paper
        labels[labels == 12] = 11  # motorcycle -> bicycle
        labels[labels == 13] = 12  # parking spot -> lanemarking
        labels[labels == 14] = 0   # road_work -> void
        labels[labels == 15] = 0   # traffic light -> void
        labels[labels == 16] = 0   # terrain -> void
        labels[labels == 17] = 11  # rider -> bicycle
        labels[labels == 18] = 8   # truck -> car
        labels[labels == 19] = 8   # bus -> car
        labels[labels == 20] = 0   # train -> void
        labels[labels == 21] = 0   # wall -> void
        labels[labels == 22] = 12  # lanemarking

        if not self.config['labels']['lanemarkings']:
            labels[labels == 12] = 0   # lanemarking -> void

        blob['labels'] = labels

        if self.config['resize']:
            blob['rgb'] = cv2.resize(blob['rgb'], (768, 384),
                                     interpolation=cv2.INTER_LINEAR)
            for m in ['depth', 'labels']:
                blob[m] = cv2.resize(blob[m], (768, 384),
                                     interpolation=cv2.INTER_NEAREST)
        return blob

    def _get_data(self, image_name=False, image=False, training_format=True):
        """Returns data for one given image number from the specified sequence."""
        if not image_name and not image:
            # one of the two should be specified
            assert False
        if image_name:
            blob = self._load_data(image_name)
        if image:
            blob = {}
            for m in image:
                blob[m] = image[m].copy()

        if training_format:
            blob = augmentate(blob,
                              scale=self.config['augmentation']['scale'],
                              crop=self.config['augmentation']['crop'],
                              hflip=self.config['augmentation']['hflip'],
                              vflip=self.config['augmentation']['vflip'],
                              gamma=self.config['augmentation']['gamma'],
                              contrast=self.config['augmentation']['contrast'],
                              brightness=self.config['augmentation']['brightness'],
                              rotate=self.config['augmentation']['rotate'],
                              shear=self.config['augmentation']['shear'])

            # Format labels into one-hot
            blob['labels'] = np.array(self.one_hot_lookup ==
                                      blob['labels'][:, :, None]).astype(int)

        # We have to add a dimension for the channels, as there is only one and the
        # dimension is omitted.
        blob['depth'] = np.expand_dims(blob['depth'], 3)

        # Force the image dimension to be multiple of 16
        h, w, _ = blob['rgb'].shape
        h_c, w_c = [d - (d % 16) for d in [h, w]]
        if h_c != h or w_c != w:
            for m in ['rgb', 'depth', 'labels']:
                blob[m] = blob[m][:h_c, :w_c, ...]

        return blob
