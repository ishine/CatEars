# NNET
fbank-test.conf
```
--num-mel-bins=40
--use-energy=false
--energy-floor=0.0
--raw-energy=true
--htk-compat=false
--use-log-fbank=true
--use-power=true
--sample-frequency=16000
--frame-length=25.0
--frame-shift=10.0
--preemphasis-coefficient=0.97
--dither=0.0  # Disable dither to make fbank matrix consistent
--window-type=hamming
--round-to-power-of-two=true
--snip-edges=true
--low-freq=20
--high-freq=8000

```

```bash
nnet3-latgen-faster --frames-per-chunk=50 --extra-left-context=0 --extra-right-context=0 --extra-left-context-initial=-1 --extra-right-context-final=-1 --minimize=false --max-active=7000 --min-active=200 --beam=15.0 --lattice-beam=8.0 --acoustic-scale=0.1 --allow-partial=true --word-symbol-table=exp/tri5a/graph/words.txt exp/nnet3/tdnn_ce/final.mdl exp/tri5a/graph/HCLG.fst "ark,s,cs:compute-fbank-feats --config=/tmp/fbank-test.conf scp:/tmp/fbank-test.scp ark:- |" "ark:|gzip -c >exp/nnet3/tdnn_ce/decode_test/lat.1.gz"
```

# CMVN

```bash
compute-fbank-feats --config=/tmp/fbank-test.conf scp:/tmp/fbank-test.scp ark:- | apply-cmvn-online 'matrix-sum scp:data/train_ce/cmvn.scp -|' ark:- ark:-|
```

# Compute WER/CER

Assuming decoder output is `hyp.txt`. For Chinese, the first step is converting word to char.

```bash
$ pocketkaldi/tool/conv_chartok.sh hyp.txt hyp.chars.txt
```

Then compute WER/CER by `compute-wer`

```bash
$ cat hyp.chars.txt | compute-wer --text --mode=present ark:exp/nnet3/tdnn/decode_test/scoring_kaldi/test_filt.chars.txt ark,p:-
```