import tensorflow as tf
from tensorflow.python.layers.layers import dropout
import numpy as np
from experiments.utils import ExperimentData
from os import path
from types import GeneratorType
import threading

from .dirichlet_fastfit import meanprecision_with_sufficient_statistic, \
                               fixedpoint_with_sufficient_statistic
#from .dirichletEstimation import findDirichletPriors
from .dirichletDifferentiation import findDirichletPriors
from .base_model import BaseModel
from xview.models.adapnet import adapnet
from xview.models.simple_fcn import encoder, decoder


def dirichlet_uncertainty_fusion(probs, conditional_params, uncertainties, prior):
    num_classes = int(probs[0].shape[3])

    # Preparation for uncertainty parameter mixture
    standard_params = np.eye(num_classes) + np.ones((num_classes, num_classes))
    standard_params = tf.reshape(standard_params.astype('float32'),
                                 [1, 1, 1, num_classes, num_classes])

    # We will collect all posteriors in this list
    log_likelihoods = []
    for i_expert in range(len(conditional_params)):
        mix = tf.reduce_mean(uncertainties[i_expert], axis=3) / \
            tf.reduce_max(uncertainties[i_expert])
        mix = tf.expand_dims(tf.expand_dims(mix, axis=-1), axis=-1)

        # The exact dirichlet params are given by the uncertainty and the standard params
        dirichlet_params = conditional_params[i_expert] * (1 - mix) +\
            mix * standard_params

        # compute p(expert output | groudn truth class x)
        conditionals = [tf.contrib.distributions.Dirichlet(dirichlet_params[..., c],
                                                           validate_args=False,
                                                           allow_nan_stats=False)
                        for c in range(num_classes)]
        log_likelihood = tf.stack(
            [conditionals[c].log_prob(1e-20 + probs[i_expert])
             for c in range(num_classes)], axis=3)
        log_likelihoods.append(log_likelihood)

    fused_likelihood = tf.reduce_sum(tf.stack(log_likelihoods, axis=0), axis=0)
    #fused_likelihood = tf.Print(
    #    fused_likelihood, [tf.reduce_any(tf.logical_or(tf.is_nan(fused_likelihood),
    #                                                   tf.is_inf(fused_likelihood)))],
    #    message='nans or infs in likelihood')
    return fused_likelihood + tf.log(1e-20 + prior)


class UncertaintyMix(BaseModel):
    """FCN implementation following DA-RNN architecture and using tf.layers."""

    def __init__(self, output_dir=None, **config):
        standard_config = {
            'learning_rate': 0.0,
        }
        standard_config.update(config)

        self.modalities = config['modalities']

        # If specified, load mesaurements of the experts.
        if 'measurement_exp' in config:
            measurements = np.load(ExperimentData(config["measurement_exp"])
                                   .get_artifact("counts.npz"))
            self.dirichlet_params = {modality: measurements[modality].astype('float32')
                                     for modality in self.modalities}
            self.class_counts = measurements['class_counts'].astype('float32')
        else:
            print('WARNING: Could not yet import measurements, you need to fit this '
                  'model first.')

        BaseModel.__init__(self, 'MixFCN', output_dir=output_dir,
                           supports_training=False, **config)

    def _build_graph(self):
        """Builds the whole network. Network is split into 2 similar pipelines with shared
        weights, one for training and one for testing."""

        num_classes = self.config['num_classes']

        # Network for testing / evaluation
        # As before, we define placeholders for the input. These here now can be fed
        # directly, e.g. with a feed_dict created by _evaluation_food
        # rgb channel
        self.test_placeholders = {}
        for modality, channels in self.config['num_channels'].items():
            self.test_placeholders[modality] = tf.placeholder(
                tf.float32, shape=[None, None, None, channels])

        def get_prob(inputs, modality, reuse=False):
            prefix = self.config['prefixes'][modality]

            if self.config['expert_model'] == 'adapnet':
                # Now we get the network output of the Adapnet expert.
                outputs = adapnet(inputs, prefix, self.config['num_units'],
                                  self.config['num_classes'], reuse=reuse)
            elif self.config['expert_model'] == 'fcn':
                outputs = encoder(inputs, prefix, self.config['num_units'],
                                  trainable=False, reuse=reuse)
                outputs.update(decoder(outputs['fused'], prefix,
                                       self.config['num_units'],
                                       self.config['num_classes'], 0.0,
                                       trainable=False, reuse=reuse))
            else:
                raise UserWarning('ERROR: Expert Model {} not found'
                                  .format(self.config['expert_model']))
            prob = tf.nn.softmax(outputs['score'])
            return prob

        def test_pipeline(inputs, modality):

            def sample_pipeline(inputs, modality, reuse=False):
                prefix = self.config['prefixes'][modality]

                # We apply dropout at the input.
                # We do want to set whole pixels to 0, therefore out noise-shape has
                # dim 1 for the channel-space:
                input_shape = tf.shape(inputs)
                noise_shape = [input_shape[0], input_shape[1], input_shape[2], 1]
                inputs = dropout(inputs, rate=self.config['dropout_rate'], training=True,
                                 noise_shape=noise_shape,
                                 name='{}_dropout'.format(prefix))
                return get_prob(inputs, modality, reuse=reuse)

            # For classification, we sample distributions with Dropout-Monte-Carlo and
            # fuse output according to variance
            samples = tf.stack([sample_pipeline(inputs, modality, reuse=(i != 0))
                                for i in range(self.config['num_samples'])], axis=4)

            mean, variance = tf.nn.moments(samples, [4])
            #variance = tf.nn.l2_normalize(variance, 3, epsilon=1e-12)

            # We get the label by passign the input without dropout
            return get_prob(inputs, modality, reuse=True), variance

        # The following part can only be build if measurements are already present.
        if hasattr(self, 'dirichlet_params'):

            probs = {}
            vars = {}
            for modality in self.modalities:
                probs[modality], vars[modality] = test_pipeline(
                    self.test_placeholders[modality], modality)

            self.probs = {modality: probs[modality] /
                          tf.reduce_sum(probs[modality], axis=3, keep_dims=True)
                          for modality in self.modalities}

            #rgb_label = tf.argmax(self.rgb_prob, 3, name='rgb_label_2d')
            #depth_label = tf.argmax(self.depth_prob, 3, name='depth_label_2d')

            #self.rgb_branch = {'label': rgb_label, 'prob': self.rgb_prob,
            #                   'var': rgb_var}
            #self.depth_branch = {'label': depth_label, 'prob': self.depth_prob,
            #                     'var': depth_var}

            # Set the Prior of the classes
            uniform_prior = 1.0 / 14
            data_prior = (self.class_counts /
                          (1e-20 + self.class_counts.sum())).astype('float32')
            if self.config['class_prior'] == 'uniform':
                # set a uniform prior for all classes
                prior = uniform_prior
            elif self.config['class_prior'] == 'data':
                prior = data_prior
            else:
                # The class_prior parameter is now considered a weight for the mixture
                # between both priors.
                weight = float(self.config['class_prior'])
                prior = weight * uniform_prior + (1 - weight) * data_prior
                prior = prior / prior.sum()

            self.fused_score = dirichlet_uncertainty_fusion(
                self.probs.values(), self.dirichlet_params.values(), vars.values(),
                prior)


            label = tf.argmax(self.fused_score, 3, name='label_2d')
            self.prediction = label

            # debugging stuff

            # compute p(expert output | groudn truth class x)

            """
            rgb_log_likelihood = tf.stack([rgb_dirichlets[c].log_prob(1e-20 + self.rgb_prob)
                                           for c in range(num_classes)], axis=3)
            depth_log_likelihood = tf.stack([depth_dirichlets[c].log_prob(1e-20 + self.depth_prob)
                                       for c in range(num_classes)], axis=3)

            rgb_log_likelihood_u = tf.stack([rgb_dirichlets[c]._log_unnormalized_prob(self.rgb_prob)
                                           for c in range(num_classes)], axis=3)
            depth_log_likelihood_u = tf.stack([depth_dirichlets[c]._log_unnormalized_prob(self.depth_prob)
                                       for c in range(num_classes)], axis=3)

            rgb_log_likelihood_n = tf.stack([rgb_dirichlets[c]._log_normalization()
                                           for c in range(num_classes)], axis=0)
            depth_log_likelihood_n = tf.stack([depth_dirichlets[c]._log_normalization()
                                       for c in range(num_classes)], axis=0)

            rgb_log_prob = tf.log(self.rgb_prob)
            depth_log_prob = tf.log(self.depth_prob)

            self.rgb_branch.update({'likelihood': rgb_log_likelihood,
                                    'u_l': rgb_log_likelihood_u,
                                    'n_l': rgb_log_likelihood_n,
                                    'log_prob': rgb_log_prob})
            self.depth_branch.update({'likelihood': depth_log_likelihood,
                                      'u_l': depth_log_likelihood_u,
                                      'n_l': depth_log_likelihood_n,
                                      'log_prob': depth_log_prob})
            """
        else:

            # Build a training pipeline for measuring the differnet classifiers
            def train_pipeline(inputs, modality, labels):
                prob = get_prob(inputs, modality)

                stacked_labels = tf.stack([labels for _ in range(num_classes)], axis=3)

                eps = 1e-10
                sufficient_statistics = [
                    tf.log(tf.where(tf.equal(stacked_labels, c), eps + prob,
                                    tf.ones_like(prob)))
                    for c in range(num_classes)]
                combined = tf.stack([tf.reduce_sum(stat, axis=[0, 1, 2])
                                     for stat in sufficient_statistics], axis=0)
                return combined

            self.train_placeholders = {}
            for modality, channels in self.config['num_channels'].items():
                self.train_placeholders[modality] = tf.placeholder(
                    tf.float32, shape=[None, None, None, channels])
            self.train_Y = tf.placeholder(tf.float32, shape=[None, None, None])
            # An input queue is defined to load the data for several batches in advance and
            # keep the gpu as busy as we can.
            # IMPORTANT: The size of this queue can grow big very easily with growing
            # batchsize, therefore do not make the queue too long, otherwise we risk getting
            # killed by the OS
            q = tf.FIFOQueue(2, [tf.float32 for _ in range(len(self.modalities) + 1)])
            queue = [self.train_Y]
            for placeholder in self.train_placeholders.values():
                queue.append(placeholder)
            self.enqueue_op = q.enqueue(queue)
            minibatches = q.dequeue()
            training_labels = minibatches[0]
            train_data = {self.modalities[i]: minibatches[i+1]
                          for i in range(len(self.modalities))}
            # The queue output does not have a defined shape, so we have to define it here to
            # be compatible with tf.layers.
            for modality in self.modalities:
                train_data[modality].set_shape([None, None, None,
                                                self.config['num_channels'][modality]])

            # This operation has to be called to close the input queue and free the space it
            # occupies in memory.
            self.close_queue_op = q.close(cancel_pending_enqueues=True)
            self.queue_is_empty_op = tf.equal(q.size(), 0)
            # To support tensorflow 1.2, we have to set this flag manually.
            self.queue_is_closed = False

            self.sufficient_statistics = {m: train_pipeline(train_data[m], m,
                                                            training_labels)
                                          for m in self.modalities}
            self.class_counts = tf.stack(
                [tf.reduce_sum(tf.cast(tf.equal(training_labels, c), tf.int64))
                 for c in range(num_classes)])

            # For compliance with base_model, we have to define a prediction outcome.
            # As we do not yet know how to do fusion, we simply take rgb.
            self.prediction = 0
            self.fused_score = 0

        # To understand what"s going on under the hood, we expose a lot of intermediate
        # results for evaluation

    def _get_sufficient_statistic(self, data):
        """Generate a sufficient statistic of the given data to fit it later to a
        dirichlet model.

        Args:
            data: the data to fit, in batch or generator format
        Returns:
            rgb and depth statistics aswell as class-counts
        """
        num_classes = self.config['num_classes']

        with self.graph.as_default():
            # store all measurements in these matrices
            counts = {m: np.zeros((num_classes, num_classes))
                      for m in self.modalities}
            class_counts = np.zeros(num_classes).astype('int64')

            # Create a thread to load data.
            coord = tf.train.Coordinator()
            t = threading.Thread(target=self._load_and_enqueue,
                                 args=(self.sess, data))
            t.start()

            queue_empty = False
            while not (queue_empty and self.queue_is_closed):
                ops = [self.class_counts]
                for m in self.modalities:
                    ops.append(self.sufficient_statistics[m])
                new_counts = self.sess.run(ops)
                class_counts += new_counts[0]
                i = 1
                for m in self.modalities:
                    counts[m] += new_counts[i]
                    i += 1
                queue_empty = self.sess.run(self.queue_is_empty_op)

            coord.join([t])
        return counts, class_counts

    def _fit_sufficient_statistic(self, counts, class_counts):
        """Fit a dirichlet model to the given sufficient statistic."""
        num_classes = self.config['num_classes']

        # Now, given the sufficient statistic, run Expectation-Maximization to get the
        # Dirichlet parameters
        def dirichlet_em(measurements):
            """Find dirichlet parameters for all class-conditional dirichlets in
            measurements."""
            params = np.ones((num_classes, num_classes)).astype('float64')

            for c in range(num_classes):
                # Average the measurements over the encoutnered class examples to get the
                # sufficient statistic.
                if class_counts[c] == 0:
                    params[:, c] = np.ones(num_classes)
                    continue
                else:
                    ss = (measurements[c, :] / class_counts[c]).astype('float64')

                # sufficient statistic of negatives
                neg_ss = (measurements.sum(0) - measurements[c, :]) / \
                    (class_counts.sum() - class_counts[c])
                print(ss)
                print(class_counts[c])

                # The prior assumption is that all class output probabilities are equally
                # likely, i.e. all concentration parameters are 1
                prior = np.ones((num_classes)).astype('float64')

                #params[:, c] = meanprecision_with_sufficient_statistic(
                #    ss, class_counts[c], num_classes, prior, maxiter=10000,
                #    tol=1e-5, delta=self.config['delta'])
                #params[:, c] = fixedpoint_with_sufficient_statistic(
                #    ss, class_counts[c], num_classes, prior, maxiter=10000,
                #    tol=1e-5, delta=self.config['delta'])
                params[:, c] = findDirichletPriors(ss, neg_ss, prior, max_iter=10000,
                                                   delta=self.config['delta'],
                                                   beta=self.config['beta'])

                print('parameters for class {}: {}'.format(
                    c, ', '.join(['{}: {:.1f}'.format(i, params[i, c])
                                  for i in range(num_classes)])))
            return params

        self.dirichlet_params = {modality: dirichlet_em(counts[modality])
                                 for modality in self.modalities}
        self.class_counts = class_counts

        if self.output_dir is not None:
            np.savez(path.join(self.output_dir, 'counts.npz'), class_counts=class_counts,
                     **self.dirichlet_params)

        # Rebuild the graph with the new measurements:
        self._initialize_graph()

    def fit(self, data, *args, **kwargs):
        """Measure the encoder outputs against the given groundtruth for the given data.

        Args:
            data: As usual, an instance of DataWrapper
        """
        modality_counts, class_counts = self._get_sufficient_statistic(data)
        print('INFO: Measurements of classifiers finished, now EM')

        self._fit_sufficient_statistic(modality_counts, class_counts)
        print("INFO: MixFCN fitted to data")

    def _evaluation_food(self, data):
        feed_dict = {self.test_placeholders[modality]: data[modality]
                     for modality in self.modalities}
        return feed_dict

    def _enqueue_batch(self, batch, sess):
        with self.graph.as_default():
            feed_dict = {self.train_Y: batch['labels']}
            for modality in self.modalities:
                feed_dict[self.train_placeholders[modality]] = batch[modality]
            sess.run(self.enqueue_op, feed_dict=feed_dict)

    def _load_and_enqueue(self, sess, data):
        """Internal handler method for the input data queue. Will run in a seperate
        thread.

        Overwritten from base_model because we only want to go once through the data.

        Args:
            sess: The current session, needs to be the same as in the main thread.
            data: The data to load. See method predict for specifications.
        """
        with self.graph.as_default():
            # We enqueue new data until it tells us to stop.
            try:
                if isinstance(data, GeneratorType):
                    for batch in data:
                        self._enqueue_batch(batch, sess)
                else:
                    self._enqueue_batch(data, sess)
            except tf.errors.CancelledError:
                print('INFO: Input queue is closed, cannot enqueue any more data.')
            sess.run(self.close_queue_op)
            self.queue_is_closed = True

    def prediction_difference(self, data):
        """Evaluate prediction of the different individual branches for the given data.
        """
        keys = self.rgb_branch.keys()
        with self.graph.as_default():
            measures = [self.prediction, self.fused_score]
            for tensors in (self.rgb_branch, self.depth_branch):
                for key in keys:
                    measures.append(tensors[key])
            outputs = self.sess.run(measures,
                                    feed_dict=self._evaluation_food(data))
        ret = {}
        ret['fused_label'] = outputs[0]
        ret['fused_score'] = outputs[1]
        i = 2
        for prefix in ('rgb', 'depth'):
            for key in keys:
                ret['{}_{}'.format(prefix, key)] = outputs[i]
                i = i + 1
        return ret

    def get_probs(self, data):
        with self.graph.as_default():
            probs = self.sess.run(self.probs.values(),
                                  feed_dict=self._evaluation_food(data))
        return probs
