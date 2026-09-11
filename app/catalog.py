"""Configured geographic and economic scope for the DENUE dataset.

Every code here comes from official INEGI catalogues:

* Municipality (alcaldía) codes: INEGI Marco Geoestadístico, entidad 09
  (Ciudad de México). https://www.inegi.org.mx/app/ageeml/
* Economic-activity codes: SCIAN México 2023 six-digit classes as exposed by
  the DENUE API ``Clase`` parameter. https://www.inegi.org.mx/scian/

The ingestion script verifies the activity label returned by DENUE for each
class code against ``official_label`` and reports any mismatch, so a wrong
mapping is caught before data is published.
"""

from dataclasses import dataclass

ENTITY_CODE = "09"
ENTITY_NAME = "Ciudad de México"


@dataclass(frozen=True, slots=True)
class Borough:
    code: str  # 3-digit INEGI municipality code within entidad 09
    name: str
    aliases: tuple[str, ...]

    @property
    def geo_code(self) -> str:
        """5-digit entidad+municipio code used by DENUE ``Cuantificar``."""
        return f"{ENTITY_CODE}{self.code}"


@dataclass(frozen=True, slots=True)
class Category:
    key: str
    label_en: str
    label_es: str
    scian_classes: tuple[str, ...]
    official_labels: tuple[str, ...]  # SCIAN 2023 class titles, one per code
    aliases: tuple[str, ...]


BOROUGHS: tuple[Borough, ...] = (
    Borough("014", "Benito Juárez", ("benito juarez", "benito juárez", "bj")),
    Borough("015", "Cuauhtémoc", ("cuauhtemoc", "cuauhtémoc", "centro")),
    Borough("016", "Miguel Hidalgo", ("miguel hidalgo", "polanco")),
    Borough("003", "Coyoacán", ("coyoacan", "coyoacán")),
    Borough("007", "Iztapalapa", ("iztapalapa",)),
)

CATEGORIES: tuple[Category, ...] = (
    Category(
        key="grocery",
        label_en="Grocery stores",
        label_es="Tiendas de abarrotes y misceláneas",
        scian_classes=("461110",),
        official_labels=("Comercio al por menor en tiendas de abarrotes, ultramarinos y misceláneas",),
        aliases=(
            "grocery",
            "groceries",
            "grocery store",
            "abarrotes",
            "miscelanea",
            "miscelánea",
            "tienditas",
        ),
    ),
    Category(
        key="convenience",
        label_en="Convenience stores",
        label_es="Minisúpers",
        scian_classes=("462112",),
        official_labels=("Comercio al por menor en minisupers",),
        aliases=(
            "convenience",
            "convenience store",
            "convenience stores",
            "minisuper",
            "minisúper",
            "minisupers",
        ),
    ),
    Category(
        key="supermarket",
        label_en="Supermarkets",
        label_es="Supermercados",
        scian_classes=("462111",),
        official_labels=("Comercio al por menor en supermercados",),
        aliases=("supermarket", "supermarkets", "supermercado", "supermercados"),
    ),
    Category(
        key="pharmacy",
        label_en="Pharmacies",
        label_es="Farmacias",
        scian_classes=("464111", "464112"),
        official_labels=("Farmacias sin minisúper", "Farmacias con minisúper"),
        aliases=("pharmacy", "pharmacies", "farmacia", "farmacias", "drugstore", "drugstores"),
    ),
    Category(
        key="bakery",
        label_en="Bakeries",
        label_es="Panaderías",
        scian_classes=("311812",),
        official_labels=("Panificación tradicional",),
        aliases=("bakery", "bakeries", "panaderia", "panadería", "panaderias", "panaderías", "bread"),
    ),
    Category(
        key="butcher",
        label_en="Butcher shops",
        label_es="Carnicerías y pollerías",
        scian_classes=("461121", "461122"),
        official_labels=("Comercio al por menor de carnes rojas", "Comercio al por menor de carne de aves"),
        aliases=(
            "butcher",
            "butchers",
            "butcher shop",
            "carniceria",
            "carnicería",
            "carnicerias",
            "polleria",
            "pollería",
            "meat",
        ),
    ),
    Category(
        key="hardware",
        label_en="Hardware stores",
        label_es="Ferreterías y tlapalerías",
        scian_classes=("467111",),
        official_labels=("Comercio al por menor en ferreterías y tlapalerías",),
        aliases=(
            "hardware",
            "hardware store",
            "hardware stores",
            "ferreteria",
            "ferretería",
            "ferreterias",
            "tlapaleria",
            "tlapalería",
        ),
    ),
    Category(
        key="department_store",
        label_en="Department stores",
        label_es="Tiendas departamentales",
        scian_classes=("462210",),
        official_labels=("Comercio al por menor en tiendas departamentales",),
        aliases=(
            "department store",
            "department stores",
            "general retail",
            "tienda departamental",
            "tiendas departamentales",
        ),
    ),
    Category(
        key="restaurant",
        label_en="Restaurants",
        label_es="Restaurantes y fondas",
        scian_classes=("722511", "722513", "722514"),
        official_labels=(
            "Restaurantes con servicio de preparación de alimentos a la carta o de comida corrida",
            "Restaurantes con servicio de preparación de antojitos",
            "Restaurantes con servicio de preparación de tacos y tortas",
        ),
        aliases=(
            "restaurant",
            "restaurants",
            "restaurante",
            "restaurantes",
            "fonda",
            "fondas",
            "taqueria",
            "taquería",
            "eateries",
            "dining",
        ),
    ),
    Category(
        key="cafe",
        label_en="Cafés",
        label_es="Cafeterías y neverías",
        scian_classes=("722515",),
        official_labels=("Cafeterías, fuentes de sodas, neverías, refresquerías y similares",),
        aliases=(
            "cafe",
            "cafes",
            "café",
            "cafés",
            "cafeteria",
            "cafetería",
            "cafeterias",
            "coffee",
            "coffee shop",
            "coffee shops",
        ),
    ),
)

# DENUE "Estrato" codes as documented for the API.
EMPLOYMENT_STRATA: dict[str, str] = {
    "1": "0 a 5 personas",
    "2": "6 a 10 personas",
    "3": "11 a 30 personas",
    "4": "31 a 50 personas",
    "5": "51 a 100 personas",
    "6": "101 a 250 personas",
    "7": "251 y más personas",
}

BOROUGHS_BY_CODE: dict[str, Borough] = {b.code: b for b in BOROUGHS}
BOROUGHS_BY_NAME: dict[str, Borough] = {b.name: b for b in BOROUGHS}
CATEGORIES_BY_KEY: dict[str, Category] = {c.key: c for c in CATEGORIES}
CATEGORY_BY_SCIAN_CLASS: dict[str, Category] = {
    code: category for category in CATEGORIES for code in category.scian_classes
}
