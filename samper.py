import math
import torch
from torch.utils.data.sampler import RandomSampler


class BatchSchedulerSampler(torch.utils.data.sampler.Sampler):
    """
    iterate over tasks and provide a random batch per task in each mini-batch
    """
    def __init__(self, dataset, batch_size, ratio):
        self.dataset = dataset
        self.batch_size = batch_size
        self.number_of_datasets = len(dataset.datasets) # 所有读取的datasets长度总和
        self.largest_dataset_size = max([len(cur_dataset) for cur_dataset in dataset.datasets])
        self.ratio = ratio
    def __len__(self):
        return self.batch_size * math.ceil(self.largest_dataset_size / self.batch_size) * len(self.dataset.datasets)

    def __iter__(self):
        samplers_list = []
        # 总的生成器
        sampler_iterators = []
        for dataset_idx in range(self.number_of_datasets):
            cur_dataset = self.dataset.datasets[dataset_idx]
            sampler = RandomSampler(cur_dataset) # 先对每个数据集的iterator进行shuffle
            samplers_list.append(sampler)
            cur_sampler_iterator = sampler.__iter__()
            sampler_iterators.append(cur_sampler_iterator)

        push_index_val = [0] + self.dataset.cumulative_sizes[:-1] # 找到每个dataset第一个数据的index
        step = self.batch_size * self.number_of_datasets # 步长为每个dataset的mini_batch_size之和
        samples_to_grab = [int(self.batch_size * r) for r in self.ratio]
        # for this case we want to get all samples in dataset, this force us to resample from the smaller datasets
        epoch_samples = self.largest_dataset_size * self.number_of_datasets # sample数量小的dataset会循环提取
        # print(epoch_samples)
        final_samples_list = []  # this is a list of indexes from the combined dataset
        for _ in range(0, epoch_samples, step):
            for i in range(self.number_of_datasets):
                # 确定迭代器
                cur_batch_sampler = sampler_iterators[i]
                cur_samples = []
                # 当前数据集迭代器的batch
                cur_samples_to_grab = samples_to_grab[i]
                for _ in range(cur_samples_to_grab):
                    try:
                        cur_sample_org = cur_batch_sampler.__next__()
                        cur_sample = cur_sample_org + push_index_val[i]
                        cur_samples.append(cur_sample)
                    except StopIteration: # 把小dataset的数据重复读取，扩充小dataset长度和最大的dataset一样
                        # got to the end of iterator - restart the iterator and continue to get samples
                        # until reaching "epoch_samples"
                        sampler_iterators[i] = samplers_list[i].__iter__()
                        cur_batch_sampler = sampler_iterators[i]
                        cur_sample_org = cur_batch_sampler.__next__()
                        cur_sample = cur_sample_org + push_index_val[i]
                        cur_samples.append(cur_sample)
                final_samples_list.extend(cur_samples)

        return iter(final_samples_list)