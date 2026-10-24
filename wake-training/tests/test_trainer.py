import argparse

from wake_training import trainer


def test_the_tflite_switch_is_off_unless_given_and_every_other_argument_is_as_declared() -> None:
    parser = argparse.ArgumentParser()
    trainer._added(parser, "--convert_to_tflite", action="store_true", default="False")  # pyright: ignore[reportPrivateUsage]
    trainer._added(parser, "--overwrite", action="store_true", default="False")  # pyright: ignore[reportPrivateUsage]
    assert vars(parser.parse_args([])) == {"convert_to_tflite": False, "overwrite": "False"}
    assert parser.parse_args(["--convert_to_tflite"]).convert_to_tflite is True
