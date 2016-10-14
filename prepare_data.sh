#!/bin/bash

#get data set

if [ "$1" == "stt" ] || [ -z "$1" ] || [ "$1" == "both" ] ; then
  echo 'Downloading speech dataset...'
  wget -P data/ http://www.openslr.org/resources/12/dev-clean.tar.gz
  echo 'Halfway done...'
  wget -P data/ http://www.openslr.org/resources/12/test-clean.tar.gz

  #unzip
  echo 'Unzipping dataset...'
  tar -zxvf data/test-clean.tar.gz LibriSpeech/
  tar -zxvf data/dev-clean.tar.gz LibriSpeech/

  mv LibriSpeech data/

  echo 'removing tar files...'
  rm data/dev-clean.tar.gz
  rm data/test-clean.tar.gz

  echo 'Done...'
fi

if [ "$1" == "lm" ] || [ -z "$1" ] || [ "$1" == "both" ] ; then
  echo "Downloading text corpus for language model..."
  
fi
