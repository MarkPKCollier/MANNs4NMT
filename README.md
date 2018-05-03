# Memory Augmented Neural Networks for Machine Translation

This repository extends the [Tensorflow neural machine translation tutorial](https://github.com/tensorflow/nmt) to create a series of models which apply memory augmented neural networks to machine translation.

We have trained Vietnamese to English and Romanian to English translation models for each of our novel architectures.

- [Usage](#usage)
- [Model 1: Neural Turing Machine Style Attention](#model-1-neural-turing-machine-style-attention)
- [Model 2: Memory Augmented Decoder](#model-2-memory-augmented-decoder)
- [Model 3: Pure MANN](#model-3-pure-mann)

Neural Turing Machine Style Attention and Memory Augmented Decoders are both extensions to the attentional encoder-decoder. We find that these extensions do not improve translation quality over the attentional encoder-decoder for the language pairs tested. The Pure MANN model is a departure from the attentional encoder-decoder. We find that the Pure MANN model performs equally to the attentional encoder-decoder for the Vietnamese to English task and ~2 BLEU worse than the attentional encoder-decoder on the Romanian to English task.

Precise results to follow...

## Usage

The following command will train a Vietnamese to English translation model using a Neural Turing Machine which recieves only the embedded source sentence as input.

You must have downloaded the IWSLT 2015 dataset to train this model (see: nmt/scripts/download_iwslt15.sh).

```
python -m nmt.nmt \
    --src=vi --tgt=en \
    --out_dir=/tmp/ref_model_en_vi_sgd_uni \
    --vocab_prefix=/path/to/iwslt/vocab \
    --train_prefix=/path/to/iwslt/train \
    --dev_prefix=/path/to/iwslt/tst2012 \
    --test_prefix=/path/to/iwslt/tst2013 \
    --attention=scaled_luong \
    --num_train_steps=14000 \
    --steps_per_stats=100 \
    --num_layers=2 \
    --num_units=512 \
    --dropout=0.3 \
    --metrics=bleu \
    --optimizer=adam \
    --learning_rate=0.001 \
    --encoder_type=bi \
    --decay_steps=1000 \
    --start_decay_step=20000 \
    --beam_width=10 \
    --share_vocab=False \
    --src_max_len=50 \
    --src_max_len_infer=50 \
    --model=model3 \
    --mann=ntm \
    --read_heads=1 \
    --write_heads=1 \
    --num_memory_locations=64 \
    --memory_unit_size=50
```

## Model 1 Neural Turing Machine Style Attention

We extend Luong attention with the ability to iterate from a Neural Turing Machine. We are motivated by the empirical observation that attention often monotonically iterates through the source sentence.

## Model 2 Memory Augmented Decoder

We add an external memory unit to the decoder of an attentional encoder-decoder. We are motivated by the successful addition of attention to the encoder-model architecture. We note that attention extends the memory capacity of the encoder, but the writable memory capacity of the decoder is still a fixed size vector. Our proposed model increases the writable memory capacity of the decoder.

![Model 2](/img/attentional_encoder_decoder_with_mann_decoder.png)

## Model 3 Pure MANN

We evaluate the use of a MANN directly for machine translation. The MANN (either a NTM or DNC) receives as input the embedded source sentence, followed by an EOS marker, after which it must output the target sentence.

![Model 3](/img/pure_mann.png)
