from pet_adoption.utils import (
    date_today_str,
)


def test_date_today_str():
    today_str = date_today_str()
    assert isinstance(today_str, str)
