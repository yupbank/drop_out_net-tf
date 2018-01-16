import tensorflow as tf

import model
import data

tf.app.flags.DEFINE_string('wmf_location', '/tmp/wmf_model',
                                   """Checkpoints for trained wmf model""")

batch_size = 20
epoch = 10
num_of_items = 100

def main(_):
    user_data = data.user_dataset()
    item_data = data.item_dataset()
    for i in xrange(epoch):
        users = user_data.batch(batch_size)
        items = item_data.batch(batch_size)
        # according to the paper we should do three type of transformations to user/item preference.
        # identity/dropout/average, while the code realised is not doning the average. 
        # and it's really non-trivial to do average, we also ommit here
         
        # we apply first half identity transorfmation
        drop_out_mask = tf.random_shuffle(tf.sequence_mask(int(drop_out*batch_size), batch_size))

        user_content = tf.stack(users.values())
        user_preference = tf.nn.embedding_lookup(wmf.user_factors, users['id'])
        item_preference = tf.nn.embedding_lookup(wmf.item_factors, items['id'])

        dropped_user_preference = tf.where(drop_out_mask, user_preference, tf.zeros_like(user_preference))
        dropped_item_preference = tf.where(drop_out_mask, item_preference, tf.zeros_like(item_preference))

        pred = model.deep_cf_model(dropped_user_preference, dropped_item_preference, user_content, item_content)
        truth = user_preference.dot(item_preference)
        loss = tf.losses.mean_squared_error(truth, pred)
        update_ops = tf.train.MomentumOptimizer(self.lr_placeholder, 0.9).minimize(self.loss)
        while True:
            _, dloss = sess.run([update_ops, loss])
            print dloss

if __name__ == "__main__":
    tf.app.run()
