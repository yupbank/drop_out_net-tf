# drop_out_net-tf
----

re-implement-drop-out-net

The drop out net is composed of two steps.

Step one, train a weighted matrix factorization model base on user/item rating/interation matrix to obtain both user preference and item preference $U_u$ and $I_i$.

code for step one is in `drop_out/train_wmf.py`

Step two, train a drop out net based on user/item preference and user/item content.

code for step two is in `drop_out/train_dropnet.py`


both models are defined in `model.py`  and data pre-processing are in `data.py`

the input are in csv format with example data `example_data/interactions.csv`, `items.csv`, `user.csv`
