import tensorflow as tf


CSV_COLUMNS = [
'id', 'jobrole', 'career_level', 'discipline_id', 'industry_id', 'country', 
'region', 'experience_n_entries_class', 'experience_years_experience', 'experience_years_in_current',
'edu_degree', 'edu_fieldofstudies', 'wtcj', 'premium'
]

CSV_COLUMN_DEFAULTS = [
    [0], [''], [0], [0], [0], [''],
    [0], [0], [0], [0], [0], [''],
    [0], [0]
    ]

CATEGORICAL_COLS = ['country']
MUTLI_VALUE_COLS = ['jobrole', 'edu_fieldofstudies']
CONTINUOUS_COLS = []

def input_parser(line):
    items = tf.decode_csv(line, CSV_COLUMN_DEFAULTS, '\t')
    features = dict(zip(CSV_COLUMNS, items))
    for cate in CATEGORICAL_COLS:
        f = features[cate]
        f = tf.one_hot(tf.string_to_hash_bucket_fast(f, 19), depth=19)
        features[cate] = tf.squeeze(f)

    for cate in CSV_COLUMNS:
        if cate in MUTLI_VALUE_COLS:
            f = features[cate]
            f = tf.string_split(tf.expand_dims(f, 0), ',').values
            f = tf.one_hot(tf.string_to_hash_bucket_fast(f, 19), depth=19)
            f = tf.reduce_sum(f, 0)
            features[cate] = f
        elif cate not in CATEGORICAL_COLS:
            features[cate] = tf.expand_dims(tf.to_float(features[cate]), -1)

    return features


def get_dataset(dataset='example_data/users.csv', batch_size=4):
    dataset = tf.contrib.data.TextLineDataset(dataset)

    dataset = dataset.skip(1).map(input_parser)

    # dataset = dataset.shuffle(buffer_size=100)
    dataset = dataset.batch(batch_size)
    return dataset


def main():
    with tf.Graph().as_default() as g:
        sess = tf.InteractiveSession()
        dataset = get_dataset()
        d = dataset.make_one_shot_iterator().get_next()
        print d
        x = tf.concat(d.values(), 1)
        try:
            while True:
                print sess.run(x).shape
        except:
            pass

if __name__ == "__main__":
    main()
