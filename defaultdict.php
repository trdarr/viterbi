<?php

/** A quick PHP port of Python's defaultdict. */
class defaultdict implements ArrayAccess {
  private $dict;
  private $default;

  public function __construct(Closure $default) {
    $this->dict = array();
    $this->default = $default;
  }

  public function offsetExists($offset) { return true; }

  public function offsetGet($offset) {
    if (isset($this->dict[$offset]))
      return $this->dict[$offset];
    else {
      $default = $this->default;
      return $default();
    }
  }

  public function offsetSet($offset, $value) {
    $this->dict[$offset] = $value;
  }

  public function offsetUnset($offset) { /* Empty. */ }

  public function keys() { return array_keys($this->dict); }
}
