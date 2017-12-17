"""
Adapted from the code produced by: https://github.com/explosion/spaCy/tree/master/examples/keras_parikh_entailment
"""

from keras.layers import *
from keras.activations import softmax
from keras.models import Model

def StaticEmbedding(embedding_matrix, trainable=False):
    """
    Static Embedding with the option to train
    
    Parameters
    ----------
    embedding_matrix: ndarray
        Embedding Matrix for the Model
    trainable: boolean
        Whether the embedding matrix should be trainable
    """
    in_dim, out_dim = embedding_matrix.shape
    embedding = Embedding(in_dim, out_dim, weights=[embedding_matrix], trainable=trainable)
    return embedding

def unchanged_shape(input_shape):
    return input_shape

def time_distributed(x, layers):
    for l in layers:
        x = TimeDistributed(l)(x)
    return x

def align(input_1, input_2):
    attention = Dot(axes=-1)([input_1, input_2])
    w_att_1 = Lambda(lambda x: softmax(x, axis=1),
                     output_shape=unchanged_shape)(attention)
    w_att_2 = Permute((2,1))(Lambda(lambda x: softmax(x, axis=2),
                             output_shape=unchanged_shape)(attention))
    in1_aligned = Dot(axes=1)([w_att_1, input_1])
    in2_aligned = Dot(axes=1)([w_att_2, input_2])
    return in1_aligned, in2_aligned

def aggregate(x1, x2, num_class, dense_dim=300, dropout_rate=0.2, activation="relu"):
    avgpool1 = GlobalAvgPool1D()(x1)
    maxpool1 = GlobalMaxPool1D()(x1)
    avgpool2 = GlobalAvgPool1D()(x2)
    maxpool2 = GlobalMaxPool1D()(x2)
    feat1 = concatenate([avgpool1, maxpool1])
    feat2 = concatenate([avgpool2, maxpool2])
    x = Concatenate()([feat1, feat2])
    x = BatchNormalization()(x)
    x = Dense(dense_dim, activation=activation)(x)
    x = Dropout(dropout_rate)(x)
    x = BatchNormalization()(x)
    x = Dense(dense_dim, activation=activation)(x)
    x = Dropout(dropout_rate)(x)
    scores = Dense(num_class, activation='sigmoid')(x)
    return scores    

def build_model(embedding_matrix, num_class=1, 
                           projection_dim=300, projection_hidden=0, projection_dropout=0.2,
                           compare_dim=500, compare_dropout=0.2,
                           dense_dim=300, dropout_rate=0.2,
                           lr=1e-3, activation='relu', maxlen=30, trainable=False, compare_layers=None):
    q1 = Input(name='q1',shape=(maxlen,))
    q2 = Input(name='q2',shape=(maxlen,))
    
    # Embedding
    encode = StaticEmbedding(embedding_matrix, trainable=trainable)
    q1_embed = encode(q1)
    q2_embed = encode(q2)
    
    # Projection
    projection_layers = []
    if projection_hidden > 0:
        projection_layers.extend([
                Dense(projection_hidden, activation=activation),
                Dropout(rate=projection_dropout),
            ])
    projection_layers.extend([
            Dense(projection_dim, activation=None),
            Dropout(rate=projection_dropout),
        ])
    q1_encoded = time_distributed(q1_embed, projection_layers)
    q2_encoded = time_distributed(q2_embed, projection_layers)
    
    # Attention
    q1_aligned, q2_aligned = align(q1_encoded, q2_encoded)    
    
    # Compare
    q1_combined = concatenate([q1_encoded, q2_aligned])
    q2_combined = concatenate([q2_encoded, q1_aligned])
    if compare_layers is None:
        compare_layers = [
            Dense(compare_dim, activation=activation),
            Dropout(compare_dropout),
            Dense(compare_dim, activation=activation),
            Dropout(compare_dropout),
        ]
    q1_compare = time_distributed(q1_combined, compare_layers)
    q2_compare = time_distributed(q2_combined, compare_layers)
    
    # Aggregate
    scores = aggregate(q1_compare, q2_compare, num_class)
    
    model = Model(inputs=[q1, q2], outputs=scores)
    return model