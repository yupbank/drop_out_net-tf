import tensorflow as tf

slim = tf.contrib.slim

def deep_cf_model(u, v, u_content, v_content, rank_out=200, model_layers=[800, 400], is_training=True):
    """
    u: user embedded by wmf model 
    v: item embedded by wmf model 
    user_content: user content info
    item_content: item content info
    rank_out: dimension of out
    """

    u_concat = tf.concat([u, u_content], 1)
    v_concat = tf.concat([v, v_content], 1)

    with slim.arg_scope([slim.fully_connected], 
                        activation_fn=tf.nn.tanh,
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=dict(decay=0.9,
                                                center=True,
                                                scale=True,
                                                is_training=is_training),
                        ):
        u_last = slim.stack(u_concat, slim.fully_connected, model_layers, scope='user_layer')
        v_last = slim.stack(v_concat, slim.fully_connected, model_layers, scope='item_layer')

    with slim.arg_scope([slim.fully_connected], 
                        activation_fn=None):

        u_emb = slim.fully_connected(u_last, rank_out, scope='user_embed')
        v_emb = slim.fully_connected(v_last, rank_out, scope='item_embed')
            
    pred = tf.multiply(u_emb, v_emb)

    return pred


def wmf(num_of_user, num_of_item, n_components):
    return tf.contrib.factorization.WALSModel(num_of_user, num_of_item, n_components)


