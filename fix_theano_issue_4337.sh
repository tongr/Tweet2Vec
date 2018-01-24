#!/usr/bin/env bash

pool_file="$(python -c 'from lasagne.layers.pool import *'  3>&1 1>&2- 2>&3- | grep File | tail -n1 | cut -d\" -f2)"
if [ -n "${pool_file}" ]; then
  echo "Theano downsample issue detected! See also: https://github.com/Theano/Theano/issues/4337"
  echo "Trying to fix issue automatically ..."
  mv "${pool_file}" "${pool_file}.bkp"
  cat "$pool_file.bkp" | sed 's|^\(.*from theano.tensor.signal import.*\)downsample\(.*\)$|\1pool\2|' | sed 's|^\(.*\)downsample.max_pool_2d\(.*\)$|\1pool.pool_2d\2|' > "$pool_file"
  pool_file_new="$(python -c 'from lasagne.layers.pool import *'  3>&1 1>&2- 2>&3- | grep File | tail -n1 | cut -d\" -f2)"
  if [ -n "${pool_file_new}" ]; then
    echo "Unable to fix issue automatically!"
    echo "Reverting ..."
    mv "${pool_file}.bkp" "${pool_file}"
  else
    echo "Success, downsample issue fixed!"
  fi
fi
