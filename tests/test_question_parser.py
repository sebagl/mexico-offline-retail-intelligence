"""Borough, category, stratum and intent detection."""

import pytest

from app.services.question_parser import fold, parse_question


@pytest.mark.parametrize(
    ("question", "boroughs"),
    [
        ("How many stores in Benito Juárez?", ("Benito Juárez",)),
        ("how many stores in benito juarez", ("Benito Juárez",)),
        ("Stores in BJ", ("Benito Juárez",)),
        ("Compare Cuauhtemoc and Miguel Hidalgo", ("Cuauhtémoc", "Miguel Hidalgo")),
        ("Compare Miguel Hidalgo and Cuauhtémoc", ("Miguel Hidalgo", "Cuauhtémoc")),
        ("Bakeries in COYOACAN", ("Coyoacán",)),
        ("Iztapalapa pharmacies", ("Iztapalapa",)),
        ("Stores in Polanco", ("Miguel Hidalgo",)),
        ("How many establishments overall?", ()),
    ],
)
def test_borough_detection(question: str, boroughs: tuple[str, ...]) -> None:
    assert parse_question(question).boroughs == boroughs


@pytest.mark.parametrize(
    ("question", "categories"),
    [
        ("Which borough has the most grocery stores?", ("grocery",)),
        ("¿Cuántas tiendas de abarrotes hay?", ("grocery",)),
        ("farmacias en Coyoacán", ("pharmacy",)),
        ("Show examples of bakeries", ("bakery",)),
        ("panaderías y carnicerías", ("bakery", "butcher")),
        ("hardware stores vs pharmacies", ("hardware", "pharmacy")),
        ("coffee shops in Cuauhtémoc", ("cafe",)),
        ("How many minisupers are there?", ("convenience",)),
        ("How many establishments?", ()),
    ],
)
def test_category_detection(question: str, categories: tuple[str, ...]) -> None:
    assert parse_question(question).categories == categories


@pytest.mark.parametrize(
    ("question", "intent"),
    [
        ("How many establishments are included in this dataset?", "count"),
        ("¿Cuántas farmacias hay en Iztapalapa?", "count"),
        ("What percentage of the selected establishments are grocery stores?", "percentage"),
        ("Which selected borough has the most grocery stores?", "ranking"),
        ("What are the most common retail categories in Benito Juárez?", "ranking"),
        ("Which categories have the lowest representation in this dataset?", "ranking"),
        ("Compare the retail composition of Cuauhtémoc and Miguel Hidalgo.", "comparison"),
        ("Cuauhtémoc vs Miguel Hidalgo pharmacies", "comparison"),
        ("Which employment-size segment is most common among pharmacies?", "ranking"),
        ("Employment-size distribution of bakeries in Coyoacán", "distribution"),
        ("Show examples of bakeries in Coyoacán.", "examples"),
        ("Tell me about pharmacies in Coyoacán", "unknown"),
        ("How do I make a chocolate cake?", "unknown"),
    ],
)
def test_intent_detection(question: str, intent: str) -> None:
    assert parse_question(question).intent == intent


def test_ranking_direction_and_axis() -> None:
    lowest = parse_question("Which categories have the lowest representation?")
    assert lowest.ascending is True
    assert lowest.rank_axis == "category"
    borough = parse_question("Which borough has the most bakeries?")
    assert borough.ascending is False
    assert borough.rank_axis == "borough"
    stratum = parse_question("Which employment-size range is most common among pharmacies?")
    assert stratum.rank_axis == "stratum"
    auto = parse_question("Which has the most grocery stores?")
    assert auto.rank_axis == "borough"


def test_stratum_detection() -> None:
    assert parse_question("How many pharmacies have 0 a 5 personas?").strata == ("1",)
    assert parse_question("stores with 251 or more employees").strata == ("7",)
    assert parse_question("How many stores?").strata == ()


def test_fold_strips_accents_and_punctuation() -> None:
    assert fold("¿Cuántas Farmacias hay en Coyoacán?") == "cuantas farmacias hay en coyoacan"
