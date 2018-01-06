import tensorflow as tf

import model

CSV_COLUMNS = ['uid', 'iid', 'action', 'time']
CSV_COLUMN_DEFAULTS = [[0], [0], [0], [0]]
DELETE_ACTION = 4


def input_parser(line):
    items = tf.decode_csv(line, CSV_COLUMN_DEFAULTS, '\t')
    return dict(zip(CSV_COLUMNS, items))


def is_delete_action(parsed):
    action = parsed['action']
    return action != DELETE_ACTION

def post_process(data):
    del data['time']
    data['action'] = tf.ones_like(data['action'])
    return data

def reduce_fn(key, records):
    return records.batch(23).map(post_process)

def get_dataset(file_name='example_data/interactions.csv'):
    dataset = tf.contrib.data.TextLineDataset(file_name)
    dataset = (
                dataset
                .skip(1)
                .map(input_parser)
                .filter(is_delete_action)
                .group_by_window(lambda r: tf.cast(r['iid'], tf.int64),
                                reduce_fn,
                                23
                               )
                )
    return dataset


def main(_):
    with tf.Graph().as_default() as g:
        sess = tf.InteractiveSession()
        dataset = get_dataset()
        d = dataset.make_one_shot_iterator().get_next()
        print d
        try:
            while True:
                print sess.run(d)
        except:
            pass
    pass


if __name__ == "__main__":
    tf.app.run()
