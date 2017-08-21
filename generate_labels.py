import pdb
import argparse
from nltk.tokenize import sent_tokenize
import time
import Queue
from threading import Thread
from collections import namedtuple
import cPickle as pkl
import glob
import numpy
from random import shuffle, sample

# Import pythonrouge package
from pythonrouge.pythonrouge import Pythonrouge
ROUGE_path = "/qydata/ywubw/download/RELEASE-1.5.5/ROUGE-1.5.5.pl"
data_path = "/qydata/ywubw/download/RELEASE-1.5.5/data"
# Input data format
input_tag = "<input>"
output_tag = "<output>"
output_sent_sep = "@li"

ExtSummary = namedtuple('ExtSummary',
                        'doc_sents summary_sents selected_ids rouge_2')
DocSummary = namedtuple('DocSummary', 'document summary extract_ids rouge_2')

# Use SSD for acceleration
temp_root = ""
select_window = 10


def pyrouge_eval(summary_sent_list, reference_sent_list, rouge):
  summary = [summary_sent_list]
  reference = [[reference_sent_list]]

  setting_file = rouge.setting(
      files=False, summary=summary, reference=reference, temp_root=temp_root)
  results = rouge.eval_rouge(
      setting_file,
      f_measure_only=True,
      ROUGE_path=ROUGE_path,
      data_path=data_path)

  return results['ROUGE-2']


def plot_hist(count, title):
  import matplotlib.pyplot as plt
  plt.hist(count, bins='auto')
  plt.title(title)
  plt.show()


def greedy_selection(doc_sent_list, ref_sent_list, rouge):
  selected = set()
  selected_sents = []
  max_rouge = 0.0
  current_candidates = range(len(doc_sent_list))

  while True:
    max_idx = -1
    id_score_list = []
    for i in current_candidates:
      if i not in selected:
        current_sents_ids = selected_sents + [(i, doc_sent_list[i])]
        current_sents = [
            t[1] for t in sorted(current_sents_ids, key=lambda s: s[0])
        ]

        rouge_score = pyrouge_eval(current_sents, ref_sent_list, rouge)
        id_score_list.append((i, rouge_score))

        if rouge_score > max_rouge:
          max_idx = i
          max_rouge = rouge_score

    if max_idx >= 0:
      selected.add(max_idx)
      selected_sents.append((max_idx, doc_sent_list[max_idx]))
      selected_sents = sorted(selected_sents, key=lambda s: s[0])
    else:
      break

    top_ids_scores = sorted(
        id_score_list, key=lambda x: x[1], reverse=True)[:select_window]
    current_candidates = [t[0] for t in top_ids_scores]

  return (sorted(list(selected)), max_rouge)


def generate_thread(doc_summary_queue, output_queue):
  rouge = Pythonrouge(
      n_gram=2,
      ROUGE_SU4=False,
      ROUGE_L=False,
      stemming=True,
      stopwords=False,
      word_level=False,
      length_limit=False,
      length=75,
      use_cf=True,
      cf=95,
      ROUGE_W=False,
      scoring_formula="average",
      resampling=False,
      samples=1000,
      favor=False,
      p=0.5)

  count = 0
  while not doc_summary_queue.empty():
    doc_sents, ref_sents = doc_summary_queue.get()
    try:
      selected, score = greedy_selection(doc_sents, ref_sents, rouge)
      # print selected, score
    except:
      print "Greedy selection error."
      continue

    result = ExtSummary(doc_sents, ref_sents, selected, score)
    output_queue.put(result)

    count += 1
    if count % 100 == 0:
      print "Processed %d documents." % count


def generate_labels(in_path, out_path, num_threads):
  input_start = len(input_tag)
  doc_summary_queue = Queue.Queue()
  output_queue = Queue.Queue()

  with open(in_path, "r") as in_file:
    # for l in in_file.readlines()[:100]: #TODO
    for l in in_file.readlines():
      input_end = l.find(output_tag)
      input_str = l[input_start:input_end].strip().decode("utf-8")
      input_sent_list = sent_tokenize(input_str)
      input_sent_list = [s.encode("utf-8") for s in input_sent_list]

      output_start = input_end + len(output_tag)
      output_str = l[output_start:].strip()
      output_sent_list = output_str.split(output_sent_sep)

      doc_summary_queue.put((input_sent_list, output_sent_list))

  start_time = time.time()
  generate_threads = []
  for i in range(num_threads):
    t = Thread(target=generate_thread, args=(doc_summary_queue, output_queue))
    t.daemon = True
    t.start()
    generate_threads.append(t)

  for t in generate_threads:
    t.join()

  results = list(output_queue.queue)
  print "Used %fs to process %d documents." % (time.time() - start_time,
                                               len(results))

  with open(out_path, "w") as f:
    pkl.dump(results, f)
    print "Results writen to %s." % out_path


def merge_labels(in_path, out_path, use_shuffle, plot):
  filelist = glob.glob(in_path)
  dataset = []
  for fn in filelist:
    with open(fn, 'r') as f:
      try:
        split_set = pkl.load(f)
      except ValueError as e:
        print "Error when reading %s" % fn
        raise e
      dataset += split_set
  print "Data size: %d" % len(dataset)

  print "Computing statistics:"
  doc_sent_lens, doc_lens, sum_lens, num_ext_ids, rouges = [], [], [], [], []
  outputs = []

  for d in dataset:
    # Log the lengths
    for s in d.doc_sents:
      doc_sent_lens.append(len(s.split()))
    doc_lens.append(len(d.doc_sents))
    sum_lens.append(len(d.summary_sents))
    num_ext_ids.append(len(d.selected_ids))
    rouges.append(d.rouge_2)
    # Convert to DocSummary object
    outputs.append(
        DocSummary(d.doc_sents, d.summary_sents, d.selected_ids, d.rouge_2))

  # Print the statistics
  print "Sentence length: mean %f stddev %f" % (numpy.mean(doc_sent_lens),
                                                numpy.std(doc_sent_lens))
  print "Document length: mean %f stddev %f" % (numpy.mean(doc_lens),
                                                numpy.std(doc_lens))
  print "Summary length: mean %f stddev %f" % (numpy.mean(sum_lens),
                                               numpy.std(sum_lens))
  print "Number of extracted ids: mean %f stddev %f" % (numpy.mean(num_ext_ids),
                                                        numpy.std(num_ext_ids))
  print "ROUGE-2: mean %f stddev %f" % (numpy.mean(rouges), numpy.std(rouges))

  # Plot the statistics
  if plot:
    plot_hist(doc_sent_lens, 'Sentence lengths')
    plot_hist(doc_lens, 'Document lengths')
    plot_hist(sum_lens, 'Summary lengths')
    plot_hist(num_ext_ids, 'No. extracted ids')
    plot_hist(rouges, 'ROUGE-2')

  if use_shuffle:  # random shuffle the data points
    print "Shuffling..."
    for _ in xrange(3):
      shuffle(outputs)

  with open(out_path, 'w') as f:
    pkl.dump(outputs, f)
    print "Merged labels written to %s" % out_path


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
  parser.add_argument('in_path', type=str, help='Path of input file.')
  parser.add_argument('out_path', type=str, help='Path of output file.')
  parser.add_argument(
      '-m', '--mode', choices=['map', 'reduce', 'sample'], default='map')
  parser.add_argument(
      '-t', '--temp_root', type=str, default='', help='Root of temp files.')
  parser.add_argument(
      '-n', '--num_threads', type=int, default=10, help='Number of threads.')
  parser.add_argument('-s', '--shuffle', action='store_true', default=False)
  parser.add_argument('-p', '--plot', action='store_true', default=False)
  parser.add_argument(
      '--num_samples', type=int, default=0, help='Number of samples drawn.')
  args = parser.parse_args()

  if args.mode == "map":  # generate labels for each split
    temp_root = args.temp_root
    generate_labels(args.in_path, args.out_path, args.num_threads)
  elif args.mode == "reduce":  # merge all labels from splits
    merge_labels(args.in_path, args.out_path, args.shuffle, args.plot)
  else:
    sample_labels(args.in_path, args.out_path, args.num_samples)
