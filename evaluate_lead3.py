""" Evaluate performance of Lead-3. """
import pdb
import argparse
import time
import re

# Import pythonrouge package
from pythonrouge import PythonROUGE
ROUGE_dir = "/qydata/ywubw/download/RELEASE-1.5.5"

# Input data format
url_tag = "<url>"
document_tag = "<doc>"
summary_tag = "<summary>"
paragraph_sep = "</p>"
sentence_sep = "</s>"
re_sent_sep = sentence_sep + "|" + paragraph_sep


def eval_rouge(in_file):
  print "Using pythonrouge package for evaluation."
  r = PythonROUGE(
      ROUGE_dir,
      n_gram=2,
      ROUGE_SU4=False,
      ROUGE_L=True,
      stemming=True,
      stopwords=False,
      word_level=False,
      length_limit=False,
      length=75,
      use_cf=True,
      cf=95,
      ROUGE_W=False,
      ROUGE_W_Weight=1.2,
      scoring_formula="average",
      resampling=False,
      samples=1000,
      favor=False,
      p=0.5)

  # pdb.set_trace()
  num_samples = 0
  summary, reference = [], []

  for l in in_file.readlines():
    doc_start = l.find(document_tag)+ len(document_tag)
    doc_end = l.find(summary_tag)
    doc_str = l[doc_start:doc_end].strip()
    doc_sent_list = re.split(re_sent_sep, doc_str)

    summary_start = doc_end + len(summary_tag)
    summary_str = l[summary_start:].strip()
    summary_sent_list = summary_str.split(sentence_sep)

    summary.append([doc_sent_list[:3]])
    reference.append([summary_sent_list])
    num_samples += 1

  start_time = time.time()
  # Evaluate ROUGE using pythonrouge package
  print r.evaluate(summary, reference)
  total_time = time.time() - start_time
  time_per_eval = total_time / num_samples
  print "Takes %f seconds to evaluate %d samples, avg %fs." % (total_time,
                                                               num_samples,
                                                               time_per_eval)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Evaluate ROUGE of Lead-3.')
  parser.add_argument('in_path', type=str, help='Path of input data file.')
  args = parser.parse_args()

  in_file = open(args.in_path, "r")
  eval_rouge(in_file)
  in_file.close()
