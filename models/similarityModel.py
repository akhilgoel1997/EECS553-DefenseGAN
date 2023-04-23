from tensorflow.keras import layers
from tensorflow_similarity.layers import MetricEmbedding
from tensorflow_similarity.losses import MultiSimilarityLoss
from tensorflow_similarity.models import SimilarityModel
from tensorflow_similarity.samplers import TFDatasetMultiShotMemorySampler
from tensorflow_similarity.visualization import viz_neigbors_imgs


sampler = TFDatasetMultiShotMemorySampler(dataset_name='mnist', classes_per_batch=10)

inputs = layers.Input(shape=(28, 28, 1))
x = layers.Rescaling(scale=1./127.5, offset=-1)(inputs)
x = layers.Conv2D(16, 3, activation='relu')(x)
x = layers.Conv2D(32, 3, activation='relu')(x)
x = layers.Conv2D(64, 3, activation='relu')(x)
x = layers.Flatten()(x)
x = layers.Dense(64, activation='relu')(x)
outputs = MetricEmbedding(64)(x)

model = SimilarityModel(inputs, outputs)

model.compile('adam', loss=MultiSimilarityLoss())
model.fit(sampler, epochs=5)

sx, sy = sampler.get_slice(0,100)
model.index(x=sx, y=sy, data=sx)

qx, qy = sampler.get_slice(3713, 1)
nns = model.single_lookup(Xgen[0, :, :, :])

viz_neigbors_imgs(Xgen[0, :, :, :], 1, nns)