#!/bin/bash
# Copyright 2012-2014  Johns Hopkins University (Author: Daniel Povey, Yenda Trmal)
# Copyright 2018       Xiaoyang Chen
# Apache 2.0


from_file=$1
to_file=$2

cat $from_file |  perl -CSDA -ane '
  {
    print $F[0];
    foreach $s (@F[1..$#F]) {
      if (($s =~ /\[.*\]/) || ($s =~ /\<.*\>/) || ($s =~ "!SIL")) {
        print " $s";
      } else {
        @chars = split "", $s;
        foreach $c (@chars) {
          print " $c";
        }
      }
    }
    print "\n";
  }' > $to_file