# Copyright 2014 Google Inc. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Script for downloading and generating question/answer pairs.
"""

import argparse
from collections import namedtuple, deque
import hashlib
from itertools import chain
from itertools import izip
from itertools import repeat
import math
from multiprocessing.pool import Pool
from multiprocessing.pool import ThreadPool
import os
import re
import sys
import time
import cchardet as chardet
from lxml import html
import requests
import socket
import pdb
import cPickle as pkl
from nltk.tokenize import sent_tokenize

datasets = ['training', 'validation', 'test']
paragraph_sep = "</p>"
sentence_sep = "</s>"
url_tag = "<url>"
document_tag = "<doc>"
summary_tag = "<summary>"

# Regular expressions
wayback_pattern = re.compile(r'web/([^/]*)/')
entity_pattern = re.compile(r'@entity\d+')
topic_patterns = {
    'cnn': re.compile('cnn.com(:80)?/\d+/(\d+/\d+/)?(\w+)/(\d+|\w+)'),
    'dailymail': re.compile('dailymail.co.uk(:80)?/(\w+)/(\w+/)*article-\d+')
}

# Story classes
AnonymizedStory = namedtuple('AnonymizedStory', 'url content highlights')
RawStory = namedtuple('RawStory', 'url html')
TokenizedStory = namedtuple('TokenizedStory', 'url tokens paragraph_starts')


class Story(namedtuple('StoryBase', 'url content highlights')):

  def ToString(self):
    return self.content + ''.join(
        ['\n\n@highlight\n\n' + highlight for highlight in self.highlights])


class QuestionContext(
    namedtuple('QuestionContextBase',
               'url context question answer anonymization_info')):

  def ToString(self):
    return '%s\n\n%s\n\n%s\n\n%s\n\n%s' % (
        self.url, self.context, self.question, self.answer, '\n'.join([
            key + ':' + value
            for key, value in self.anonymization_info.iteritems()
        ]))


def ReadUrls(filename):
  """Reads a list of URLs.

  Args:
    filename: The filename containing the URLs.

  Returns:
    A list of URLs.
  """

  with open(filename) as f:
    return [line.strip('\n') for line in f]


def Hashhex(s):
  """Returns a heximal formated SHA1 hash of the input string.

  Args:
    s: The string to hash.

  Returns:
    A heximal formatted hash of the input string.
  """

  h = hashlib.sha1()
  h.update(s)
  return h.hexdigest()


def ReadDownloadedUrl(url, corpus):
  """Reads a downloaded URL from disk.

  Args:
    url: The URL to read.
    corpus: The corpus the URL belongs to.

  Returns:
    The content of the URL.
  """

  try:
    with open('%s/downloads/%s.html' % (corpus, Hashhex(url))) as f:
      return f.read()
  except IOError:
    return None


def ParseHtml(story, corpus):
  """Parses the HTML of a news story.

  Args:
    story: The raw Story to be parsed.
    corpus: Either 'cnn' or 'dailymail'.

  Returns:
    A Story containing URL, paragraphs and highlights.
  """

  parser = html.HTMLParser(encoding=chardet.detect(story.html)['encoding'])
  tree = html.document_fromstring(story.html, parser=parser)

  # Elements to delete.
  delete_selectors = {
      'cnn': [
          '//blockquote[contains(@class, "twitter-tweet")]',
          '//blockquote[contains(@class, "instagram-media")]'
      ],
      'dailymail': [
          '//blockquote[contains(@class, "twitter-tweet")]',
          '//blockquote[contains(@class, "instagram-media")]'
      ]
  }

  # Paragraph exclusions: ads, links, bylines, comments
  cnn_exclude = (
      'not(ancestor::*[contains(@class, "metadata")])'
      ' and not(ancestor::*[contains(@class, "pullquote")])'
      ' and not(ancestor::*[contains(@class, "SandboxRoot")])'
      ' and not(ancestor::*[contains(@class, "twitter-tweet")])'
      ' and not(ancestor::div[contains(@class, "cnnStoryElementBox")])'
      ' and not(contains(@class, "cnnTopics"))'
      ' and not(descendant::*[starts-with(text(), "Read:")])'
      ' and not(descendant::*[starts-with(text(), "READ:")])'
      ' and not(descendant::*[starts-with(text(), "Join us at")])'
      ' and not(descendant::*[starts-with(text(), "Join us on")])'
      ' and not(descendant::*[starts-with(text(), "Read CNNOpinion")])'
      ' and not(descendant::*[contains(text(), "@CNNOpinion")])'
      ' and not(descendant-or-self::*[starts-with(text(), "Follow us")])'
      ' and not(descendant::*[starts-with(text(), "MORE:")])'
      ' and not(descendant::*[starts-with(text(), "SPOILER ALERT:")])')

  dm_exclude = ('not(ancestor::*[contains(@id,"reader-comments")])'
                ' and not(contains(@class, "byline-plain"))'
                ' and not(contains(@class, "byline-section"))'
                ' and not(contains(@class, "count-number"))'
                ' and not(contains(@class, "count-text"))'
                ' and not(contains(@class, "video-item-title"))'
                ' and not(ancestor::*[contains(@class, "column-content")])'
                ' and not(ancestor::iframe)')

  paragraph_selectors = {
      'cnn': [
          '//div[contains(@class, "cnnContentContainer")]//p[%s]' % cnn_exclude,
          '//div[contains(@class, "l-container")]//p[%s]' % cnn_exclude,
          '//div[contains(@class, "cnn_strycntntlft")]//p[%s]' % cnn_exclude
      ],
      'dailymail':
      ['//div[contains(@class, "article-text")]//p[%s]' % dm_exclude]
  }

  # Highlight exclusions.
  he = ('not(contains(@class, "cnnHiliteHeader"))'
        ' and not(descendant::*[starts-with(text(), "Next Article in")])')
  highlight_selectors = {
      'cnn': [
          '//*[contains(@class, "el__storyhighlights__list")]//li[%s]' % he,
          '//*[contains(@class, "cnnStryHghLght")]//li[%s]' % he,
          '//*[@id="cnnHeaderRightCol"]//li[%s]' % he
      ],
      'dailymail': ['//h1/following-sibling::ul//li']
  }

  def ExtractText(selector):
    """Extracts a list of paragraphs given a XPath selector.

    Args:
      selector: A XPath selector to find the paragraphs.

    Returns:
      A list of raw text paragraphs with leading and trailing whitespace.
    """

    xpaths = map(tree.xpath, selector)
    elements = list(chain.from_iterable(xpaths))
    paragraphs = [e.text_content().encode('utf-8') for e in elements]

    # Remove editorial notes, etc.
    if corpus == 'cnn' and len(paragraphs) >= 2 and '(CNN)' in paragraphs[1]:
      paragraphs.pop(0)

    paragraphs = map(str.strip, paragraphs)
    paragraphs = [s for s in paragraphs if s and not str.isspace(s)]

    return paragraphs

  for selector in delete_selectors[corpus]:
    for bad in tree.xpath(selector):
      bad.getparent().remove(bad)

  paragraphs = ExtractText(paragraph_selectors[corpus])
  highlights = ExtractText(highlight_selectors[corpus])

  content = '\n\n'.join(paragraphs)

  return Story(story.url, content, highlights)


def WriteStory(story, corpus):
  """Writes a news story to disk.

  Args:
    story: The news story to write.
    corpus: The corpus the news story belongs to.
  """

  story_string = story.ToString()
  url_hash = Hashhex(story.url)

  with open('%s/stories/%s.story' % (corpus, url_hash), 'w') as f:
    f.write(story_string)


def LoadTokenMapping(filename):
  """Loads a token mapping from the given filename.

  Args:
    filename: The filename containing the token mapping.

  Returns:
    A list of (start, end) where start and
    end (inclusive) are offsets into the content for a token. The list is
    sorted.
  """

  mapping = []

  with open(filename) as f:
    line = f.readline().strip()

    for token_mapping in line.split(';'):
      if not token_mapping:
        continue

      start, length = token_mapping.split(',')

      mapping.append((int(start), int(start) + int(length)))

    mapping.sort(key=lambda x: x[1])  # Sort by start.

  return mapping


def Tokenize(story, corpus):
  """Tokenizes a news story.

  Args:
    story: The Story.
    corpus: The corpus of the news story.

  Returns:
    A TokenizedStory containing the URL and the tokens or None if no token
    mapping was found for the URL.
  """

  s = story.ToString()
  url_hash = Hashhex(story.url)

  # Get start indices of paragraphs/highlights
  paragraph_starts = deque([m.start() for m in re.finditer('\n\n', s)])

  mapping_filename = '%s/tokens/%s.txt' % (corpus, url_hash)
  if not os.path.exists(mapping_filename):
    return None
  mapping = LoadTokenMapping(mapping_filename)

  tokens = []
  p_start_tok_idx = [0]  # start token indices of each paragraph
  for i, (start, end) in enumerate(mapping):
    tokens.append(s[start:end + 1])
    if paragraph_starts and end > paragraph_starts[0]:
      p_start_tok_idx.append(i)
      paragraph_starts.popleft()
      while paragraph_starts and end > paragraph_starts[0]:
        paragraph_starts.popleft()

  return TokenizedStory(story.url, tokens, p_start_tok_idx)


def LoadEntityMapping(filename):
  """Loads an entity mapping from the given filename.

  Args:
    filename: The filename containing the entity mapping.

  Returns:
    A list of (entity_index, start, end)
    where start and end (inclusive) are token offsets for an entity. The list
    is sorted.
  """

  mapping = []

  with open(filename) as f:
    line = f.readline().strip()

    for entity_mapping in line.split(';'):
      if not entity_mapping:
        continue

      entity_index, start, end = entity_mapping.split(',')

      mapping.append((int(entity_index), int(start), int(end)))

    mapping.sort(key=lambda x: x[2])  # Sort by start.

  return mapping


def Anonymize(tokenized_story, corpus):
  """Anonymizes a tokenized news story.

  Args:
    tokenized_story: A TokenizedStory.
    corpus: The corpus of the tokenized news story.

  Returns:
    A Story containing the URL, anonymized content and anonymized highlights or
    None if no entity mapping exists for the news story.
  """
  url_hash = Hashhex(tokenized_story.url)

  mapping_filename = '%s/entities/%s.txt' % (corpus, url_hash)
  if not os.path.exists(mapping_filename):
    return None

  mapping = LoadEntityMapping(mapping_filename)

  mapping_index = 0
  mapping_len = len(mapping)

  new_tokens = []

  i = 0
  while i < len(tokenized_story.tokens):
    if mapping_index < mapping_len and mapping[mapping_index][1] == i:
      entity_index, start, end = mapping[mapping_index]
      entity_tokens = [
          t + '@ent' for t in tokenized_story.tokens[start:end + 1]
      ]
      new_tokens += entity_tokens

      mapping_index += 1
      i = end + 1
    else:
      new_tokens.append(tokenized_story.tokens[i])
      i += 1

  # parts = ' '.join(new_tokens).split(' @ highlight ')
  # content = parts[0]
  # highlights = parts[1:]

  new_tokenized_story = TokenizedStory(tokenized_story.url, new_tokens,
                                       tokenized_story.paragraph_starts)

  # return Story(tokenized_story.url, content, highlights)
  return Tokens2Story(new_tokenized_story)


def ExtractTopic(url, corpus):
  try:
    topic = re.findall(topic_patterns[corpus], url)[0]
    if corpus == 'cnn':
      new_topic = topic[2].lower()
    else:
      new_topic = topic[1].lower()
    return new_topic
  except:
    return 'unknown'


def Tokens2Story(tokenized_story):
  """Generates a list of question/answer pairs given an anonymized news story.

  One question/answer pair is generated for each anonymized entity appearing in
  the question.

  Args:
    tokenized_story: The anonymized news story.

  Returns:
    A list of QuestionContext containing questions and answers.
  """

  paragraph_starts = tokenized_story.paragraph_starts
  paragraph_tokens = []
  for i in xrange(len(paragraph_starts) - 1):
    start = paragraph_starts[i]
    end = paragraph_starts[i + 1]
    paragraph_tokens.append(tokenized_story.tokens[start:end])
  paragraph_tokens.append(tokenized_story.tokens[paragraph_starts[-1]:])

  # parts = ' '.join(tokenized_story.tokens).split(' @ highlight ')
  hl_idx = [
      i for i, x in enumerate(paragraph_tokens) if x == ['@', 'highlight']
  ]
  content_paragraphs = paragraph_tokens[:hl_idx[0]]
  highlights_tokens = []
  for idx in hl_idx:
    try:
      highlights_tokens.append(paragraph_tokens[idx + 1])
    except Exception as e:
      print e, hl_idx
      # print "\n".join([" ".join(p) for p in paragraph_tokens])
      break

  # Concat the tokens
  paragraph_str_list = []
  for p in content_paragraphs:
    p_str = " ".join(p).strip().decode("utf-8")
    p_sent_list = sent_tokenize(p_str)
    p_sent_list = [s.encode("utf-8") for s in p_sent_list]
    new_p_str = sentence_sep.join(p_sent_list)
    paragraph_str_list.append(new_p_str)
    
  content = paragraph_sep.join(paragraph_str_list)
  highlights = [" ".join(h) for h in highlights_tokens]

  return Story(tokenized_story.url, content, highlights)


def WriteAllStories(story_list, corpus, dataset, output_dir):
  """Writes a question/answer pair to disk.

  Args:
    story_list: The list of all stories in dataset.
    corpus: The corpus the question/answer belongs to.
    dataset: One of 'training', 'validation' and 'test'.
    output_dir: The directory to write output files.
  """

  story_list = [s for s in story_list if s]  # ignore Nones
  with open('%s/%s/all.%s.pkl' % (output_dir, corpus, dataset), 'w') as f:
    pkl.dump(story_list, f)

  topic_dict = {}
  for story in story_list:
    topic = ExtractTopic(story.url, corpus)
    if topic not in topic_dict:
      topic_dict[topic] = []

    story_str = " ".join([
        url_tag, story.url, document_tag, story.content, summary_tag,
        sentence_sep.join(story.highlights)
    ])
    topic_dict[topic].append(story_str)

  all_stories = []
  for topic in topic_dict.keys():
    with open('%s/%s/%s.%s' % (output_dir, corpus, dataset, topic), 'w') as f:
      f.write('\n'.join(topic_dict[topic]) + '\n')
      all_stories += topic_dict[topic]

  with open('%s/%s/all.%s' % (output_dir, corpus, dataset), 'w') as f:
    f.write('\n'.join(all_stories) + '\n')


def GenerateMapper(t):
  """Reads an URL from disk and returns a list of question/answer pairs.

  Args:
    t: a tuple (url, corpus).

  Returns:
    A list of QuestionContext containing a question and an answer.
  """

  url, corpus, anonymize = t
  story_html = ReadDownloadedUrl(url, corpus)
  if not story_html:
    return None

  raw_story = RawStory(url, story_html)

  story = ParseHtml(raw_story, corpus)
  tokenized = Tokenize(story, corpus)

  if not tokenized:
    return None

  if anonymize:
    final_story = Anonymize(tokenized, corpus)
  else:
    final_story = Tokens2Story(tokenized)

  return final_story


def GenerateMode(corpus, output_dir, anonymize):
  for dataset in datasets:
    print 'Generating doc/summary pairs for %s %s set.' % (corpus, dataset)

    urls_filename = '%s/wayback_%s_urls.txt' % (corpus, dataset)
    urls = ReadUrls(urls_filename)

    p = Pool()
    story_list = p.map(GenerateMapper,
                       izip(urls, repeat(corpus), repeat(anonymize)))
    p.close()
    p.join()
    WriteAllStories(story_list, corpus, dataset, output_dir)


def main():
  parser = argparse.ArgumentParser(
      description='Generates document/summary pairs')
  parser.add_argument(
      '--outdir',
      type=str,
      default='output',
      help='Output directory for generate mode.')
  parser.add_argument('--corpus', choices=['cnn', 'dailymail'], default='cnn')
  parser.add_argument(
      '--anonymize',
      action='store_true',
      default=False,
      help='Whether to anonymize the entities.')

  args = parser.parse_args()

  stories_dir = '%s/stories' % args.corpus
  if not os.path.exists(stories_dir):
    os.mkdir(stories_dir)

  downloads_dir = '%s/downloads' % args.corpus
  if not os.path.exists(downloads_dir):
    os.mkdir(downloads_dir)

  output_dir = '%s/%s' % (args.outdir, args.corpus)
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  GenerateMode(args.corpus, args.outdir, args.anonymize)


if __name__ == '__main__':
  main()
