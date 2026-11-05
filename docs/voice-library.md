# Voices from Kyutai's voice library

Hands can speak in any voice in Kyutai's public voice library, `kyutai/tts-voices` on Hugging
Face, as well as the built-in voices. A library voice is named by its path there, without the
`.wav`: for example `vctk/p225_023_enhanced`. A bare name such as `Zed` is refused, since every
library voice is in a folder.

To find, hear, and choose one:

1. **Find.** Ask hands to search the library for words in a path, such as "find voices in the
   library with p225". It uses `find_voices`, which returns up to 50 names. On the phone's talk
   page (`hands phone`), the gallery searches the library too, and each match can be played and
   chosen there.
2. **Hear.** Ask hands to let you hear some. `hear_voices` says a line in each, five at a time
   unless you ask for more. Hearing a voice does not choose it.
3. **Choose.** Tell hands which one to use. `use_voice` keeps it in `~/.hands/voice`, and what it
   says next is in that voice, across restarts.

The first use of a library voice fetches two things from Hugging Face: the cloning model that
Pocket TTS needs to speak in a voice given as audio, and that voice's audio prompt. Both are kept
in the Hugging Face cache. Choosing a voice that is not cached needs the network.

## The Hugging Face token

The fetch takes a Hugging Face token. hands reads it from the `HF_TOKEN` environment variable if
it is set, and otherwise from the macOS keychain item named `HF_TOKEN`. To store one there:

```sh
security add-generic-password -s HF_TOKEN -a "$USER" -w
```

Leave off the token after `-w`, so `security` prompts for it and it stays out of your shell
history. hands never logs the token or writes it to a file, and it does not pass it to the brain
or to the terminals it opens for sessions.

## When a voice cannot be used

When a library voice cannot be used, hands says why when you choose it:

- **The library has no such voice.** The path is misspelled, or the file is not in the library.
- **The token was refused.** The Hugging Face token is wrong or expired. The hub answers a refused
  token the same way it answers a missing repository, so hands names both possibilities.
- **Hugging Face cannot be reached.** A choice that needs the network fails until it is back.

## Starting with a kept library voice

A kept library voice is checked again when hands starts. If it cannot be fetched then, hands
refuses to start and names the voice. To recover, choose another voice, or remove `~/.hands/voice`
to go back to Charles.

Only one library voice has been checked end to end, with a live encode of
`vctk/p225_023_enhanced`. A file that downloads but cannot be encoded passes the choice, and the
start then fails without naming it, so remove `~/.hands/voice` if that happens.
