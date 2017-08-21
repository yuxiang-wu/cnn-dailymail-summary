import pdb
import argparse
from collections import namedtuple
import cPickle as pkl
from random import shuffle, sample

DocSummary = namedtuple('DocSummary', 'document summary extract_ids rouge_2')


def merge_labels(cnn_path, dm_path, out_path, use_shuffle):
  with open(cnn_path, 'r') as f:
    cnn_dataset = pkl.load(f)
  print "CNN data size: %d" % len(cnn_dataset)

  with open(dm_path, 'r') as f:
    dm_dataset = pkl.load(f)
  print "Daily Mail data size: %d" % len(dm_dataset)

  merged_dataset = cnn_dataset + dm_dataset

  if use_shuffle:  # random shuffle the data points
    print "Shuffling..."
    for _ in xrange(3):
      shuffle(merged_dataset)

  print "Merged data size: %d" % len(merged_dataset)
  with open(out_path, 'w') as f:
    pkl.dump(merged_dataset, f)
    print "Merged dataset written to %s" % out_path


def sample_labels(in_path, out_path, num_samples):
  with open(in_path, 'r') as f:
    dataset = pkl.load(f)
  print "Data size: %d" % len(dataset)

  sampled_dataset = sample(dataset, num_samples)
  print "Sampled %d data instances." % len(sampled_dataset)

  with open(out_path, 'w') as f:
    pkl.dump(sampled_dataset, f)
    print "Sampled dataset written to %s" % out_path


if __name__ == "__main__":
  parser = argparse.ArgumentParser(
      description='Generate labels for extractive summarization.')
  parser.add_argument('cnn_path', type=str, help='Path of CNN dataset file.')
  parser.add_argument(
      'dailymail_path', type=str, help='Path of Daily Mail data file.')
  parser.add_argument('output_path', type=str, help='Path of output file.')
  parser.add_argument('-s', '--shuffle', action='store_true', default=False)
  args = parser.parse_args()

  merge_labels(args.cnn_path, args.dailymail_path, args.output_path,
               args.shuffle)
