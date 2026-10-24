# Training a wake word

hands listens for one of openWakeWord's own phrases unless you give it a model of your own, and this repo
trains one on the Mac. With the 17 GB corpus already on disk, a word takes about 25 minutes and 1.5–1.7 GB.
The two words trained so far, Hey Hands and Okay Hands, hear their phrase on 94.7% and 94.6% of held-out clips,
and neither woke once in 10.7 hours of speech, music, and noise.

## Train a word

From the repo root:

```sh
uv run --project wake-training train-wake-word "Hey Hands"
```

A phrase is words of letters separated by single spaces; an apostrophe may sit inside a word. `--work` is where
the corpus, each word's clips and features, and the run logs go (default `~/wake/train`). `--to` is where the
model goes (default `~/wake`).

The run ends by printing two lines for the `[talk]` table of `~/.hands/config.toml`:

```toml
wake_word = "Hey Hands"
wake_word_model = "/Users/<you>/wake/hey_hands.onnx"
```

The pipeline never writes `config.toml` itself. Add the lines, then restart hands to take up the model.

## What a run does

The stages run in this order:

1. `corpus`, once for each asset every word shares, fetches it the first time: openWakeWord's ACAV100M training
   features (~16 GB) and its ~10.7-hour validation features, the MIT room impulse responses, 4 AudioSet shards
   decoded to 16 kHz WAVs, and piper-sample-generator's source at a pinned commit with its LibriTTS-R voice model.
2. `feature_models` fetches the melspectrogram and embedding models the trainer computes features with.
3. `recipe` writes the trainer's YAML.
4. `generate_clips` makes the word's clips.
5. `augment_clips` adds background noise and room sound to them and computes their features.
6. `train_model` runs openWakeWord's own `train.py`, unchanged. What the Mac needs done differently from Linux
   is in the docstring of `wake-training/src/wake_training/trainer.py`.
7. `evaluate` scores the model, as described under [How good the models are](#how-good-the-models-are).
8. `install` puts the ONNX model in `--to`, in place of any model there of the same name.

Each stage writes one JSON line, with its seconds and what it made, to `<work>/runs/<name>.jsonl`, and prints the
same line as it ends. Every time below is read from those logs.

A rerun resumes. Fetched assets are kept, and so are the clips or features of a trainer step that finished. A step
that was cut short starts over, and training and evaluation run again.

### Where a word's clips live, and why they pile up

A word's clips and features live in a run directory named for the word plus a digest of what they are made from:
the phrase, its near-miss phrases, the sample counts, the background and room data, and the voice generator. For
Hey Hands that is `words/hey_hands-c24f5cb5`. Change any of those and the next run makes a fresh directory and
generates every clip again. Change only how the model is trained (steps, width, weights) and the clips are reused.

Old run directories are never deleted for you, and each holds about 1.5–1.7 GB. Remove the ones you no longer
need by hand.

### Near misses

Each word is trained against its parts said alone, and against its last words after other lead-ins: "hey",
"okay", "my", "the", "your", and "these". For Hey Hands the near misses are "hands", "hey", "my hands", "okay
hands", "the hands", "these hands", and "your hands". They make up a quarter of the negative clips, each near
miss as often as the others, in the held-out clips as in the training ones.

Without them, each model learned "…hands". The Hey Hands model woke on 246 of 300 "okay hands" clips, and Okay
Hands woke on 277 of 300 "hey hands" clips. That matters because people say "hands" constantly when they talk about
hands. With near misses in training, those counts fell to 10 of 300 and 26 of 300.

## What it costs

Measured on an Apple M4 Max with 128 GB.

**Disk.** The corpus takes 17 GB, once: the features 16 GB, AudioSet 612 MB, piper 198 MB, and the impulse
responses 8.5 MB. Each word then takes about 1.5–1.7 GB.

**Time**, with the corpus already present:

| stage | one word, alone | each word, two trained at once |
| --- | --- | --- |
| `generate_clips` | 734 s | ~1,000 s |
| `augment_clips` | 214 s | ~400 s |
| `train_model` | 186 s | ~250 s |
| `evaluate` | 370 s | ~360 s |
| whole run | about 25 min | ~34 min for both |

Fetching the corpus's three smaller assets took 115 s. The 16 GB of features were fetched before the run log
existed, so how long they take to download is not measured.

## How good the models are

`evaluate` scores a model with openWakeWord's runtime `Model`, the one hands listens with, at hands' threshold
of 0.5. These are the shipped models:

| word | recall on 2,000 held-out clips | near-miss false accepts (2,000 clips) | false wakes in 10.7 h of speech, music, noise |
| --- | --- | --- | --- |
| Hey Hands | 94.7% | 1.45% | 0 |
| Okay Hands | 94.6% | 1.25% | 0 |

Through hands' own `WakeWord` class, streaming 300 held-out clips, Hey Hands woke on 296 of 300 "hey hands" and 10
of 300 "okay hands"; Okay Hands woke on 286 of 300 "okay hands" and 26 of 300 "hey hands".

For scale, the same false-wake measure on openWakeWord's own models gives 0.47 per hour for Hey Jarvis (hands'
default), 0.56 for Hey Mycroft, and 2.7 for Alexa.

False wakes are counted the way hands wakes. A score at the threshold is one wake. The model is then deaf for the
5 frames that openWakeWord scores 0 after a reset, as hands resets it after each wake. hands scores the frames after
those on a window still partly filled with noise from the reset; the count scores them on clean windows, so each of
them can count as a wake. openWakeWord's trainer
prints numbers of its own that disagree with these, and both differences are in how they are counted. Its "False
Positives per Hour" counts every window over the threshold and assumes 11.3 hours, so it reads higher. Its
"Recall" is taken on augmented (noisy, reverberant) clips, so it reads far lower, around 0.32, than the clean
held-out recall above.

Two limits on these numbers:

- The trainer chooses its checkpoints on the same validation features the false-wake count uses. That count
  compares models with each other; it is not a measure of how often hands will wake by mistake in your room.
- Every positive clip is a synthetic voice. Recall on a real voice in a real room is not measured here.

## Tuning so far

The recipe starts from openWakeWord's example and doubles two settings: the model width, from 32 to 64, and the
cap on negative weight, from 1,500 to 3,000. On Hey Hands, before near misses were added, that took false wakes
from 1.1 per hour to 0.37, for recall going from ~98% to ~97%. Doubling the training steps to 100,000 made it
worse: 1.7 per hour.

Two runs of one recipe can differ by more than any of those changes. Retrained from the same clips, Hey Hands came
out at 91.4% recall and 0.28 false wakes per hour, and Okay Hands at 87.1% recall, on the same held-out clips the
shipped models score 94.7% and 94.6% on with no false wakes. Train a word more than once and keep the best model:
each run installs over the model in `--to`, so give each run its own `--to` and compare their `evaluate` lines.
