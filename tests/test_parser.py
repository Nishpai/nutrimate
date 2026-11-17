"""The chat parser must always produce JSON that passes the schema."""

import json

import jsonschema
import pytest

from nutrimate.parser import REQUEST_SCHEMA, parse, to_constraints
from nutrimate.constraints import Constraints

MESSAGES = [
    "I want a high-protein vegetarian dinner under 600 calories. Avoid peanuts and suggest something Indian.",
    "show me a 30 g protein vegetarian lunch under 500 kcal",
    "quick vegan breakfast, no soy or sesame, gluten-free",
    "I'm allergic to shellfish and don't like mushrooms, Thai dinner in 45 minutes",
    "something under 700",
    "hello",
    "",
    "eggetarian snack with 20g protein",
    "a 99999 calorie dinner",  # out of range number must not break the schema
]


@pytest.mark.parametrize("msg", MESSAGES)
def test_parser_output_is_schema_valid(msg):
    result = parse(msg, use_llm=False)
    jsonschema.validate(result["request"], REQUEST_SCHEMA)
    json.dumps(result)  # serialisable


def test_example_from_brief():
    r = parse(MESSAGES[0], use_llm=False)["request"]
    assert r["diet"] == "vegetarian"
    assert r["max_calories"] == 600
    assert r["cuisine"] == "Indian"
    assert r["meal_type"] == "main"
    assert r["excluded_ingredients"] == ["peanut"]
    assert r["prefer_high_protein"] is True


def test_ambiguous_number_asks_follow_up():
    r = parse("something under 700", use_llm=False)
    assert r["follow_up"] and r["request"]["max_calories"] is None


def test_chat_cannot_remove_profile_allergen():
    base = Constraints(diet="vegan", exclude=["peanut"])
    c = to_constraints(parse("any dinner is fine, chicken ok", use_llm=False)["request"], base)
    assert c.diet == "vegan" and "peanut" in c.exclude


def test_bad_llm_output_is_rejected():
    bad = {"diet": "carnivore", "max_calories": -5}
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(bad, REQUEST_SCHEMA)
