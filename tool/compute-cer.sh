#!/bin/bash

tool_dir=`dirname $0`
ref_file=$1
hyp_file=$2

hyp_chars_file=/tmp/${hyp_file}.chars.txt

# Convert to character
$tool_dir/conv_chartok.sh $hyp_file $hyp_chars_file || exit 22
compute-wer --text --mode=present ark:$ref_file ark:$hyp_chars_file
