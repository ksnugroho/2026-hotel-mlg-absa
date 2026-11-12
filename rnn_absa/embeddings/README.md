# Embeddings

Place the fastText Indonesian model here before running experiments.

## Download

```bash
cd rnn_absa/embeddings/
wget --no-check-certificate https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.id.300.bin.gz
gunzip cc.id.300.bin.gz
```

Expected file: `cc.id.300.bin` (~4.2 GB)

The file is excluded from git via .gitignore.
