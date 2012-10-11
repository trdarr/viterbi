<?php

require_once 'defaultdict.php';
require_once 'set.php';

/**
 * An implementation of the Viterbi algorithm for POS tagging as
 *   an assignment for UT's LIN 350 "Natural Language Procssing."
 *   (See: http://nlp-s11.utcompling.com/assignments/hmm-tagging)
 * Python implementation and PHP translation by Thomas Darr <trdarr@gmail.com>.
 */
class ViterbiException extends Exception {}

class ViterbiTagger {
  /** string */
  private $train_file, $input_file;

  /** defaultdict(int) */
  private $tokens, $unique_tokens;
  private $tags, $unique_tags;
  private $emissions;

  /** set */
  private $token_dict, $tag_dict;

  /** defaultdict(set) */
  private $token_dicts, $tag_dicts;

  public function __construct($train_file=null, $input_file=null) {
    $this->train_file = $train_file;
    $this->input_file = $input_file;
  }

  /** Trains the tagger on a string of token/tag pairs. */
  public function train($train_file=null) {
    if (is_null($train_file))
      if (is_null($this->train_file))
        throw new ViterbiException('Viterbi needs a training file.');
      else $train_file = $this->train_file;
    else $this->train_file = $train_file;

    if (!file_exists($train_file) || !is_readable($train_file))
      throw new ViterbiException('Training file does not exist.');

    $int = function() { return 0; };
    $this->tokens = new defaultdict($int);         // Token counts.
    $this->tags = new defaultdict($int);           // Tag counts.
    $this->emissions = new defaultdict($int);      // Emission counts.
    $this->unique_tokens = new defaultdict($int);  // Unique token counts.
    $this->unique_tags = new defaultdict($int);    // Unique tag counts

    $x = null;
    $file = new SplFileObject($train_file);
    while (!$file->eof()) {
      $y = $file->fgets();
      if (!is_null($x))
        $this->count($x, $y);
      $x = $y;
    }

    // Token dictionary and default tag dictionary. (???)
    $this->token_dict = new set();
    foreach ($this->tokens->keys() as $token) {
      if (false === strpos($token, '/') && $token !== '')
        $this->token_dict->add($token);
    }

    $this->tag_dict = new set();
    foreach ($this->tags->keys() as $tag) {
      if (false === strpos($tag, '/') && !in_array($tag, array('', '###')))
        $this->tag_dict->add($tag);
    }

    // Token dictionaries for each known word.
    $this->tag_dicts = new defaultdict(function() { return new set(); });
    foreach ($this->emissions->keys() as $emission) {
      list($tag, $word) = explode('/', $emission);
      $set = $this->tag_dicts[$word]->add($tag);
      $this->tag_dicts[$word] = $set;
    }

    $this->trans_prob = array();
    foreach ($this->tags->keys() as $tag) {
      if (false !== strpos($tag, '/'))
        $this->trans_prob[$tag] = $this->p_tt($tag);
    }

    $this->emiss_prob = array();
    foreach ($this->emissions->keys() as $emission) {
      $this->emiss_prob[$emission] = $this->p_tw($emission);
    }

    return $this;
  }

  /** Counts a tag/token bigram. */
  private function count($x, $y) {
    $x = explode('/', trim($x));  // ['When', 'W'].
    $y = explode('/', trim($y));  // ['such', 'J'].

    if (count($x) !== count($y))
      return;  // Count correctly, please.

    // The total number of tokens/tags.
    $this->tokens[''] += 1;

    // Unigram token count, e.g., p(such).
    $this->tokens[$y[0]] += 1;
    if ($this->tokens[$y[0]] === 1)
      $this->unique_tokens[$y[0]] += 1;
    else if ($this->tokens[$y[0]] === 2)
      $this->unique_tokens[$y[0]] -= 1;

    // Unigram tag counts, e.g., p(J).
    $this->tags[$y[1]] += 1;
    if ($this->tags[$y[1]] === 1)
      $this->unique_tags[$y[1]] += 1;
    else if ($this->tags[$y[1]] === 2)
      $this->unique_tags[$y[1]] -= 1;

    // Bigram counts for tag/tag sequence, e.g., p(J|W).
    $tag_key = "{$x[1]}/{$y[1]}";
    $this->tags[$tag_key] += 1;

    // Emission counts for token/tag pair, e.g., p(such|J).
    $emission_key = "{$y[1]}/{$y[0]}";
    $this->emissions[$emission_key] += 1;
  }

  /** Computes smoothed tag transition probability. */
  private function p_tt($tag) {
    list($t1, $t2) = explode('/', $tag);

    $lambda = isset($this->unique_tags[$t1])
      ? $this->unique_tags[$t1]
      : 1e-100;
    $backoff = ((float) $this->tags[$t2])
      / ($this->tokens[''] - 1);

    $num = $this->tags[$tag] + $lambda * $backoff;
    $den = $this->tags[$t1] + $lambda;
    return log($num / $den);
  }

  /** Computes smoothed token emission probability. */
  private function p_tw($emission) {
    list($tag, $token) = explode('/', $emission);

    $lambda = isset($this->unique_tokens[$token])
      ? $this->unique_tokens[$token]
      : 1e-100;
    $backoff = ($this->tokens[$token] + 1.0)
      / ($this->tokens[''] + $this->token_dict->size());

    $num = $this->emissions[$emission] + $lambda * $backoff;
    $den = $this->tags[$tag] + $lambda;
    return log($num / $den);
  }

  public function tag($input_file=null) {
    return $this;
  }

  public function score() {
    return $this;
  }
}

$vt = new ViterbiTagger('entrain', 'entest');
$vt->train()->tag()->score();
