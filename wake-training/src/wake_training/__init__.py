"""Train a wake word for hands. One phrase goes in; an openWakeWord ONNX model comes out, which hands hears as
`[talk] wake_word` and `wake_word_model` in its config.toml. hands does not import this package, and this package
does not import hands.
"""
