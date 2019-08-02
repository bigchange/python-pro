# https://github.com/brightmart/bert_language_understanding/blob/master/model/multi_head_attention.py

import tensorflow as tf

import fire

def get_mask(batch_size, sequence_length):
    lower_triangle = tf.matrix_band_part(
        tf.ones([sequence_length, sequence_length]), -1, 0)
    result = -1e9*(1.0-lower_triangle)
    print("get_mask==>result:%d", result)
    return result

def main():
    fire.Fire()
    # we can try to use: fire.Fire(Example) to see the diff in CLI


if __name__ == '__main__':
    main()


