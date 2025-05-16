from app.utils import calculate_bmi

def test_calculate_bmi():
    assert calculate_bmi(180, 75) == round(75 / (1.8 ** 2), 2)
