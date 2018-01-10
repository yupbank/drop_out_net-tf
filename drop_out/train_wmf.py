import tensorflow as tf

import model

CSV_COLUMNS = ['uid', 'iid', 'action', 'time']
CSV_COLUMN_DEFAULTS = [[0], [0], [0], [0]]
DELETE_ACTION = 4
NUM_OF_USER = 1000
NUM_OF_ITEM = 400 
EMBEDDING_DIMENSION = 10

def input_parser(line):
    items = tf.decode_csv(line, CSV_COLUMN_DEFAULTS, '\t')
    return dict(zip(CSV_COLUMNS, items))

def prepare_input(parsed):
    return {
            WALSMatrixFactorization.INPUT_ROWS: parsed['uid'],
            WALSMatrixFactorization.INPUT_COLS: parsed['iid']
            }, None

def is_delete_action(parsed):
    action = parsed['action']
    return action != DELETE_ACTION

def post_process(data):
    del data['time']
    data['action'] = tf.ones_like(data['action'])
    return data

def reduce_fn(key, records):
    return records.batch(NUM_OF_ITEM).map(post_process)

def get_dataset(file_name='example_data/interactions.csv'):
    dataset = tf.contrib.data.TextLineDataset(file_name)
    dataset = (
                dataset
                .skip(1)
                .map(input_parser)
                .filter(is_delete_action)
                .group_by_window(lambda r: tf.cast(r['iid'], tf.int64),
                                reduce_fn,
                                NUM_OF_ITEM)
                )
    return dataset

def wmf(num_user=NUM_OF_USER, num_item=NUM_OF_ITEM, embedding_dimension=EMBEDDING_DIMENSION):
    return tf.contrib.factorization.WALSModel(num_user, num_item, embedding_dimension)

def main(_):
    with tf.Graph().as_default() as g:
        sess = tf.InteractiveSession()
        dataset = get_dataset()
        data = dataset.make_one_shot_iterator().get_next()
        model = wmf()
        model_init_op = model.initialize_op
        col_update_prep_gramian_op = model.col_update_prep_gramian_op
        row_update_prep_gramian_op = model.row_update_prep_gramian_op
        worker_init_op = model.worker_init
        init_col_update_op = model.initialize_col_update_op
        init_row_update_op = model.initialize_row_update_op
        sess.run([model_init_op])
        worker_init_op.run(session=sess)
        row_update_prep_gramian_op.run(session=sess)
        init_row_update_op.run(session=sess)
        col_update_prep_gramian_op.run(session=sess)
        init_col_update_op.run(session=sess)
        try:
            while True:
                values = tf.concat([tf.expand_dims(data['uid'], 1), tf.expand_dims(data['iid'], 1)], 1)
                sp_input = tf.SparseTensor(tf.cast(values, tf.int64), tf.ones_like(data['uid'], dtype=tf.float32), [NUM_OF_USER, NUM_OF_ITEM])
                new_col_factor, update_op, unregularized_loss, regularization, sum_weights = model.update_col_factors(sp_input)
                print sess.run([update_op, unregularized_loss])
        except Exception as e:
            pass
if __name__ == "__main__":
    tf.app.run()
