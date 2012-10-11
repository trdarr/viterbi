<?php
/**
 * A quick PHP port of Python's set.
 * By Thomas Darr <trdarr@gmail.com>.
 */
class set {
  private $dict;
  public function __construct() { $this->dict = array(); }
  public function __toString() { return json_encode(array_keys($this->dict)); }

  // Modify the set.
  public function add($value) { $this->dict[$value] = true; return $this; }
  public function remove($value) { unset($this->dict[$value]); return $this; }

  // Query the set.
  public function contains($value) { return isset($this->dict[$value]); }
  public function size() { return sizeof(array_keys($this->dict)); }
}

