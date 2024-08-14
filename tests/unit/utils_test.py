from pet_adoption.utils import (
    date_today_str,
)

# def test_get_project_root():
#     project_root = get_project_root()
#     assert project_root == Path("/home/ubuntu/pet-adoption")


# def test_get_artifact_path():
#     artifact_path = get_artifact_path()
#     assert artifact_path == Path("/home/ubuntu/pet-adoption/artifacts")


def test_date_today_str():
    today_str = date_today_str()
    assert isinstance(today_str, str)
