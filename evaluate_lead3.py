""" Evaluate performance of Lead-3. """
import pdb
import argparse
from nltk.tokenize import sent_tokenize
import time

# Import pythonrouge package
from pythonrouge import PythonROUGE
ROUGE_dir = "/qydata/ywubw/download/RELEASE-1.5.5"
# Input data format
input_tag = "<input>"
output_tag = "<output>"
output_sent_sep = "@li"


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
      use_cf=False,
      cf=95,
      ROUGE_W=False,
      ROUGE_W_Weight=1.2,
      scoring_formula="average",
      resampling=True,
      samples=1000,
      favor=False,
      p=0.5)

  # pdb.set_trace()
  input_start = len(input_tag)
  num_samples = 0
  summary, reference = [], []

  for l in in_file.readlines():
    input_end = l.find(output_tag)
    output_start = input_end + len(output_tag)
    input_str = l[input_start:input_end].strip().decode("utf-8")
    output_str = l[output_start:].strip().decode("utf-8")

    input_sent_list = sent_tokenize(input_str)
    input_sent_list = [s.encode("utf-8") for s in input_sent_list]
    output_sent_list = output_str.split(output_sent_sep)
    output_sent_list = [s.encode("utf-8") for s in output_sent_list]

    summary.append([input_sent_list[:3]])
    reference.append([output_sent_list])
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
