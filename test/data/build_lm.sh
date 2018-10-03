# This script is used to train a sample LM from lm.train.txt and convert to
# pasco FST format. 

cat lm.train.txt | sed 's/ /\n/g' | sort | uniq | awk 'BEGIN { print "<eps> 0"; a=1; } { if ($0 != "")  {print $0" "a; a++;} }' > lm.words.txt
irstlm build-lm -i lm.train.txt -f 2 -o lm.arpa
gunzip -c lm.arpa.gz | arpa2fst --read-symbol-table=lm.words.txt - G.fst
python ../../tool/convert_fstfmt.py G.fst G.pfst
