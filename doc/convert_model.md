# Convert FST

```bash
```


# Convert CMVN Statistics

Before covnverting, we need to compute global CMVN statistics. Goto s5 directory and run

```bash
$ compute-cmvn-stats --binary=false scp:data/train_ce/feats.scp cmvn_stat.global
compute-cmvn-stats --binary=false scp:data/train_ce/feats.scp cmvn_stat.global 
LOG (compute-cmvn-stats[5.4.149~1-9b23b]:main():compute-cmvn-stats.cc:168) Wrote global CMVN stats to cmvn_stat.global
LOG (compute-cmvn-stats[5.4.149~1-9b23b]:main():compute-cmvn-stats.cc:171) Done accumulating CMVN stats for 120098 utterances; 0 had errors.
```

Then convert to pocketkaldi format

```bash
$ python3 $POCKETKALDI_DIR/tool/convert_cmvn_stats.py cmvn_stat.global cmvn_stat.bin
```

# Convert Transition Model

Here we will convert Kaldi transcrition model to pocketkaldi format. Actually, we only need the map from transition-id to pdf-id.

First, extract the map from final.mdl

```bash
$ $POCKETKALDI_DIR/build/extract_id2pdf exp/nnet3/tdnn_ce/final.mdl > t2pdf.txt
/home/ling0322/pocketkaldi/build/extract_id2pdf exp/nnet3/tdnn_ce/final.mdl
```

Then convert to pocketkaldi format

```bash
$ python3 $POCKETKALDI_DIR/tool/convert_trans.py t2pdf.txt t2pdf.bin
num_pdfs = XXXX
```

convert_trans.py will output number of pdf in the model in `num_pdfs = XXXX`. Remember this value, it will be used in config file.


# Copy Vocabulary

The vocabulary file from Kaldi (aka words.txt) could be used directly in pasco decoder. We can just copy it to model dir. 

