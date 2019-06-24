ACLOCAL_AMFLAGS = -I m4
AUTOMAKE_OPTIONS = gnu subdir-objects foreign
AM_CFLAGS = 
AM_CXXFLAGS = -I$(top_srcdir)/src -I$(top_srcdir)/src/openfst/include -I$(top_srcdir)/src/gemmlowp -g -fPIC -std=c++11
AM_LDFLAGS = -pthread 
LIBS = -lm

bin_PROGRAMS = pocketkaldi
pocketkaldi_SOURCES = src/main.cc

lib_LIBRARIES = libpocketkaldi.a libgemmlowp.a libfst.a

libpocketkaldi_a_SOURCES = src/util.cc \
                           src/fst.cc \
                           src/matrix.cc \
                           src/pcm_reader.cc \
                           src/decoder.cc \
                           src/srfft.cc \
                           src/fbank.cc \
                           src/strlcpy.cc \
                           src/cmvn.cc \
                           src/nnet.cc \
                           src/am.cc \
                           src/vector.cc \
                           src/symbol_table.cc \
                           src/ce_stt.cc \
                           src/hashtable.cc \
                           src/configuration.cc
pocketkaldi_LDADD = libpocketkaldi.a libgemmlowp.a libfst.a -lstdc++ -lopenblas

libgemmlowp_a_SOURCES = src/gemmlowp/eight_bit_int_gemm/eight_bit_int_gemm.cc
libgemmlowp_a_CXXFLAGS = -I$(top_srcdir)/src -I$(top_srcdir)/src/gemmlowp -g -fPIC -std=c++11

libfst_a_SOURCES = src/openfst/lib/compat.cc \
                   src/openfst/lib/flags.cc \
                   src/openfst/lib/fst.cc \
                   src/openfst/lib/fst-types.cc \
                   src/openfst/lib/mapped-file.cc \
                   src/openfst/lib/properties.cc \
                   src/openfst/lib/symbol-table.cc \
                   src/openfst/lib/symbol-table-ops.cc \
                   src/openfst/lib/weight.cc \
                   src/openfst/lib/util.cc
libfst_a_CXXFLAGS = -I$(top_srcdir)/src -I$(top_srcdir)/src/openfst/include -g -fPIC -std=c++11



if ENABLE_TOOLS
    AM_CXXFLAGS += -I$(KALDI_ROOT)/src -I$(KALDI_ROOT)/tools/openfst-1.6.5/include
    bin_PROGRAMS += compute_fbank extract_id2pdf
    compute_fbank_SOURCES = tool/compute_fbank.cc
    compute_fbank_LDADD = $(KALDI_ROOT)/src/util/kaldi-util.a \
                          $(KALDI_ROOT)/src/feat/kaldi-feat.a \
                          $(KALDI_ROOT)/src/matrix/kaldi-matrix.a \
                          $(KALDI_ROOT)/src/base/kaldi-base.a \
                          $(ATLASLIBS) \
                          libpocketkaldi.a

    extract_id2pdf_SOURCES = tool/extract_id2pdf.cc
    extract_id2pdf_LDADD = $(KALDI_ROOT)/src/util/kaldi-util.a \
                           $(KALDI_ROOT)/src/hmm/kaldi-hmm.a \
                           $(KALDI_ROOT)/src/matrix/kaldi-matrix.a \
                           $(KALDI_ROOT)/src/base/kaldi-base.a \
                           $(ATLASLIBS) \
                           libpocketkaldi.a
endif

TESTS = fst_test \
        srfft_test \
        fbank_test \
        nnet_test \
        hashtable_test \
        configuration_test \
        pool_test \
        gemm_test

check_PROGRAMS = fst_test \
                 srfft_test \
                 fbank_test \
                 nnet_test \
                 hashtable_test \
                 configuration_test \
                 pool_test \
                 gemm_test

configuration_test_SOURCES = test/configuration_test.cc
configuration_test_CXXFLAGS = $(AM_CXXFLAGS) -std=c++11 -I$(top_srcdir)/src -DTESTDIR=\"$(top_srcdir)/test/\"
configuration_test_LDADD = libpocketkaldi.a libgemmlowp.a -lopenblas

hashtable_test_SOURCES = test/hashtable_test.cc
hashtable_test_CXXFLAGS = $(AM_CXXFLAGS) -std=c++11 -I$(top_srcdir)/src
hashtable_test_LDADD = libpocketkaldi.a libgemmlowp.a -lopenblas

fst_test_SOURCES = test/fst_test.cc
fst_test_CXXFLAGS = $(AM_CXXFLAGS) -std=c++11 -I$(top_srcdir)/src -DTESTDIR=\"$(top_srcdir)/test/\"
fst_test_LDADD = libpocketkaldi.a libgemmlowp.a -lopenblas

srfft_test_SOURCES = test/srfft_test.cc
srfft_test_CXXFLAGS = $(AM_CXXFLAGS) -std=c++11 -I$(top_srcdir)/src
srfft_test_LDADD = libpocketkaldi.a libgemmlowp.a -lopenblas

fbank_test_SOURCES = test/fbank_test.cc
fbank_test_CXXFLAGS = $(AM_CXXFLAGS) -std=c++11 -I$(top_srcdir)/src -DTESTDIR=\"$(top_srcdir)/test/\"
fbank_test_LDADD = libpocketkaldi.a libgemmlowp.a -lopenblas

nnet_test_SOURCES = test/nnet_test.cc
nnet_test_CXXFLAGS = $(AM_CXXFLAGS) -std=c++11 -I$(top_srcdir)/src -DTESTDIR=\"$(top_srcdir)/test/\"
nnet_test_LDADD = libpocketkaldi.a libgemmlowp.a -lopenblas

gemm_test_SOURCES = test/gemm_test.cc
gemm_test_CXXFLAGS = $(AM_CXXFLAGS) -std=c++11 -I$(top_srcdir)/src -DTESTDIR=\"$(top_srcdir)/test/\"
gemm_test_LDADD = libpocketkaldi.a libgemmlowp.a -lopenblas

pool_test_SOURCES = test/pool_test.cc
pool_test_CXXFLAGS = $(AM_CXXFLAGS) -std=c++11 -I$(top_srcdir)/src
pool_test_LDADD = libpocketkaldi.a libgemmlowp.a -lopenblas

if ENABLE_TOOLS
    TESTS_ENVIRONMENT = export testdir=$(top_srcdir)/test && export kaldiroot=$(KALDI_ROOT) &&
    TESTS += test/test_compute_fbank.sh
endif