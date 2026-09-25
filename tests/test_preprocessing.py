from router.preprocessing import normalize


def test_lowercase_and_yo():
    assert normalize("Приёмная КОМИССИЯ") == "приемная комиссия"


def test_punctuation_and_spaces():
    assert normalize("  Здравствуйте!!!   Где,  расписание?? ") == "здравствуйте где расписание"


def test_keeps_latin_and_digits():
    assert normalize("Ошибка 500 в LMS") == "ошибка 500 в lms"


def test_only_punctuation_becomes_empty():
    assert normalize("?!... 🙂") == ""
