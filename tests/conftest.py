import pytest

from rossmann.spark import get_spark


@pytest.fixture(scope="session")
def spark():
    session = get_spark(app_name="rossmann-tests", cores="2", driver_memory="2g")
    yield session
    session.stop()
