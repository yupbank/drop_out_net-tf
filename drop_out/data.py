import tensorflow as tf
from functools import partial


USER_CSV_COLUMNS = [
'id', 'jobrole', 'career_level', 'discipline_id', 'industry_id', 'country', 
'region', 'experience_n_entries_class', 'experience_years_experience', 'experience_years_in_current',
'edu_degree', 'edu_fieldofstudies', 'wtcj', 'premium'
]

USER_CSV_COLUMN_DEFAULTS = [
    [0], [''], [0], [0], [0], [''],
    [0], [0], [0], [0], [0], [''],
    [0], [0]
    ]
USER_CATEGORICAL_COLS = {'country': 19}
USER_MUTLI_VALUE_COLS = {
                        'jobrole':100, 
                         'edu_fieldofstudies':100
                         }

ITEM_CSV_COLUMNS = [
'id', 'title', 'career_level', 'discipline_id', 'industry_id', 'country', 
'is_payed', 
'region', 'latitude', 'longitude', 'employment', 'tags', 'created_at'
]
ITEM_CSV_COLUMN_DEFAULTS = [
    [0], [''], [0], [0], [0], [''],
    [0], 
    [0], [0.0], [0.0], [0], [''], [0]
]
ITEM_CATEGORICAL_COLS = {'country': 19}
ITEM_MUTLI_VALUE_COLS = {
                        'title':100, 
                        'tags':100
                        }


def input_parser(csv_columns, csv_column_defaults, categorical_cols, mutli_value_cols, line):
    items = tf.decode_csv(line, csv_column_defaults, '\t', na_value='null')
    features = dict(zip(csv_columns, items))
    for cate in categorical_cols:
        f = features[cate]
        f = tf.one_hot(tf.string_to_hash_bucket_fast(f, categorical_cols[cate]), depth=categorical_cols[cate])
        features[cate] = tf.squeeze(f)

    for cate in csv_columns:
        if cate in mutli_value_cols:
            f = features[cate]
            f = tf.string_split(tf.expand_dims(f, 0), ',').values
            f = tf.one_hot(tf.string_to_hash_bucket_fast(f, mutli_value_cols[cate]), depth=mutli_value_cols[cate])
            f = tf.reduce_sum(f, 0)
            features[cate] = f
        elif cate not in categorical_cols:
            features[cate] = tf.expand_dims(tf.to_float(features[cate]), -1)

    return features

user_input_parser = partial(input_parser, USER_CSV_COLUMNS, USER_CSV_COLUMN_DEFAULTS, USER_CATEGORICAL_COLS, USER_MUTLI_VALUE_COLS)
item_input_parser = partial(input_parser, ITEM_CSV_COLUMNS, ITEM_CSV_COLUMN_DEFAULTS, ITEM_CATEGORICAL_COLS, ITEM_MUTLI_VALUE_COLS)

def get_dataset(dataset='example_data/users.csv', input_parser=user_input_parser, batch_size=4):
    dataset = tf.data.TextLineDataset(dataset)

    dataset = dataset.skip(1).map(input_parser)

    # dataset = dataset.shuffle(buffer_size=100)
    dataset = dataset.batch(batch_size)
    return dataset

user_dataset = get_dataset()
item_dataset = get_dataset('example_data/items.csv', item_input_parser)

def main():
    with tf.Graph().as_default() as g:
        sess = tf.InteractiveSession()
        u = user_dataset.make_one_shot_iterator().get_next()
        dus = tf.concat(u.values(), 1)
        i = item_dataset.make_one_shot_iterator().get_next()
        dis = tf.concat(i.values(), 1)
        try:
            while True:
                print 'user', sess.run(dus).shape
        except:
            pass
        print '----'
        try:
            while True:
                print 'item', sess.run(dis).shape
        except:
            pass

if __name__ == "__main__":
    main()
