#!/bin/bash
# Copyright 2010-2012 Microsoft Corporation
#           2012-2013 Johns Hopkins University (Author: Daniel Povey)
#           2018-2019 ling0322<ling032x@gmail.com>
# Apache 2.0

# This script creates a fully expanded decoding graph (HCLG) that represents
# all the language-model, pronunciation dictionary (lexicon), context-dependency,
# and HMM structure in our model.  The output is a Finite State Transducer
# that has word-ids on the output, and pdf-ids on the input (these are indexes
# that resolve to Gaussian Mixture Models).
# See
#  http://kaldi-asr.org/doc/graph_recipe_test.html
# (this is compiled from this repository using Doxygen,
# the source for this part is in src/doc/graph_recipe_test.dox)

. path.sh

if [[ -z $POCKETKALDI_ROOT ]]; then
  echo "env $$POCKETKALDI_ROOT is empty"
  exit 22
fi

# Clear pasco_graph folder
[[ -d pasco_graph ]] && rm -r pasco_graph

lm=$1
lm_arpa=lm.arpa
lm_arpa_gz=lm.order1.arpa.gz
lm_1order_arpa=lm.1order.arpa
lm_1order_bin=lm.1order.bin

# Prune the language model with order = 1
gunzip -c $lm | python3 prune_lm.py | gzip > $lm_arpa_gz

# G compilation, check LG composition
utils/format_lm.sh data/lang $lm_arpa_gz \
    data/local/dict/lexicon.txt pasco_graph || exit 1;

# Make HCLG
mkdir pasco_graph/lang_test
cp -r data/lang_test/phones data/lang_test/L_disambig.fst data/lang_test/words.txt\
      data/lang_test/L.fst data/lang_test/phones.txt pasco_graph/lang_test || exit 22
cp pasco_graph/G.fst pasco_graph/lang_test/G.fst || exit 22
utils/mkgraph.sh pasco_graph/lang_test exp/tri5a pasco_graph || exit 22

# Make G'
gunzip -c $lm | python3 $POCKETKALDI_ROOT/tool/prune_lm.py > $lm_1order_arpa || exit 22
python3 $POCKETKALDI_ROOT/tool/convert_unigram.py $lm_1order_arpa pasco_graph/lang_test/words.txt lm.1order.bin
gunzip -c $lm | arpa2fst --read-symbol-table=pasco_graph/lang_test/words.txt - G_raw.fst
python $POCKETKALDI_ROOT/tool/convert_fstfmt.py G_raw.fst G.pfst
