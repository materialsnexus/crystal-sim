from fireworks import LaunchPad
from pymongo.synchronous.mongo_client import MongoClient

from crystal import LOCAL_LAUNCHPAD_YAML


def test_local_launchpad() -> None:
    launchpad = LaunchPad.from_file(LOCAL_LAUNCHPAD_YAML.as_posix())
    assert isinstance(launchpad, LaunchPad)
    assert launchpad.host == "localhost"
    assert launchpad.name == "local_launchpad"
    assert isinstance(launchpad.connection, MongoClient)
