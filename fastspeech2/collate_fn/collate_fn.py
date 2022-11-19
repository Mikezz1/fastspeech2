from fastspeech2.datasets.lj_speech import reprocess_tensor
import numpy as np


class collate_fn_tensor:
    def __init__(self, train_config):
        self.batch_expand_size = train_config.batch_expand_size

    def __call__(self, batch):
        len_arr = np.array([d["text"].size(0) for d in batch])
        index_arr = np.argsort(-len_arr)
        batchsize = len(batch)
        real_batchsize = batchsize // self.batch_expand_size

        cut_list = list()
        for i in range(self.batch_expand_size):
            cut_list.append(index_arr[i*real_batchsize:(i+1)*real_batchsize])

        output = list()
        for i in range(self.batch_expand_size):
            output.append(reprocess_tensor(batch, cut_list[i]))

        return output
