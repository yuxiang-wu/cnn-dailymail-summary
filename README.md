# README for CNN/Daily Mail summarization dataset

This folder contains the CNN/Daily Mails dataset for summarization. It is based on https://github.com/deepmind/rc-data

## How to run?

1. Run generate_summaries.py to preprocess the downloaded dataset in 'cnn' and 'dailymail' folder. Output files are written to 'output_v*/no_entity'.
2. Run generate_labels.py to generate extractive summarization labels. Output files are written to 'output_v*/split_labels'.
3. Run generate_labels.py to merge the split labels into one file. Output files are written to 'output_v*/merged_labels'.

## Versions

1. output_v1: both no_entity and with_entity versions of raw data is available. Sentences in the document are segmented only by NLTK.
2. output_v2: only no_entity version of raw data is available. The document are first segmented by  paragraph tag \<p> in html. Then the paragraphs are further segmented into sentences using NLTK.