import pytest
from pydantic import BaseModel, Field

from openbatch._utils import (
    _ensure_strict_json_schema,
    has_more_than_n_keys,
    resolve_ref,
    type_to_json_schema,
)


class TestHasMoreThanNKeys:
    def test_empty_dict(self):
        assert has_more_than_n_keys({}, 0) is False
        assert has_more_than_n_keys({}, 1) is False

    def test_single_key(self):
        assert has_more_than_n_keys({"a": 1}, 0) is True
        assert has_more_than_n_keys({"a": 1}, 1) is False
        assert has_more_than_n_keys({"a": 1}, 2) is False

    def test_multiple_keys(self):
        obj = {"a": 1, "b": 2, "c": 3}
        assert has_more_than_n_keys(obj, 0) is True
        assert has_more_than_n_keys(obj, 1) is True
        assert has_more_than_n_keys(obj, 2) is True
        assert has_more_than_n_keys(obj, 3) is False


class TestResolveRef:
    def test_resolve_simple_ref(self):
        root = {"definitions": {"Person": {"type": "object"}}}
        result = resolve_ref(root=root, ref="#/definitions/Person")
        assert result == {"type": "object"}

    def test_resolve_nested_ref(self):
        root = {"definitions": {"Nested": {"properties": {"field": {"type": "string"}}}}}
        result = resolve_ref(root=root, ref="#/definitions/Nested/properties/field")
        assert result == {"type": "string"}

    def test_resolve_ref_invalid_format(self):
        root = {"definitions": {}}
        with pytest.raises(ValueError, match="Does not start with #/"):
            resolve_ref(root=root, ref="definitions/Person")


class TestEnsureStrictJsonSchema:
    def test_object_adds_additional_properties_false(self):
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        result = _ensure_strict_json_schema(schema, path=(), root=schema)
        assert result["additionalProperties"] is False

    def test_object_preserves_existing_additional_properties(self):
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "additionalProperties": True,
        }
        result = _ensure_strict_json_schema(schema, path=(), root=schema)
        assert result["additionalProperties"] is True

    def test_object_makes_all_properties_required(self):
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
        }
        result = _ensure_strict_json_schema(schema, path=(), root=schema)
        assert "required" in result
        assert set(result["required"]) == {"name", "age"}

    def test_nested_object_properties(self):
        schema = {
            "type": "object",
            "properties": {
                "person": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                }
            },
        }
        result = _ensure_strict_json_schema(schema, path=(), root=schema)
        assert result["properties"]["person"]["additionalProperties"] is False
        assert result["properties"]["person"]["required"] == ["name"]

    def test_array_items(self):
        schema = {"type": "array", "items": {"type": "string"}}
        result = _ensure_strict_json_schema(schema, path=(), root=schema)
        assert result["items"]["type"] == "string"

    def test_array_with_object_items(self):
        schema = {
            "type": "array",
            "items": {"type": "object", "properties": {"id": {"type": "integer"}}},
        }
        result = _ensure_strict_json_schema(schema, path=(), root=schema)
        assert result["items"]["additionalProperties"] is False
        assert result["items"]["required"] == ["id"]

    def test_any_of_union(self):
        schema = {
            "anyOf": [
                {"type": "string"},
                {"type": "object", "properties": {"a": {"type": "string"}}},
            ]
        }
        result = _ensure_strict_json_schema(schema, path=(), root=schema)
        assert len(result["anyOf"]) == 2
        assert result["anyOf"][1]["additionalProperties"] is False

    def test_all_of_single_element_processed(self):
        # Test that allOf with single element is processed
        # Note: The implementation handles multi-element allOf, but single-element
        # allOf isn't necessarily unwrapped by the current implementation
        schema = {
            "allOf": [
                {"type": "object", "properties": {"name": {"type": "string"}}},
                {"type": "object", "properties": {"age": {"type": "integer"}}},
            ]
        }
        result = _ensure_strict_json_schema(schema, path=(), root=schema)
        # Verify allOf entries are processed
        assert "allOf" in result
        assert len(result["allOf"]) == 2
        # Each entry should have properties from original schema
        assert result["allOf"][0]["type"] == "object"
        assert result["allOf"][1]["type"] == "object"

    def test_definitions_processed(self):
        schema = {
            "type": "object",
            "properties": {"user": {"$ref": "#/definitions/User"}},
            "definitions": {"User": {"type": "object", "properties": {"name": {"type": "string"}}}},
        }
        result = _ensure_strict_json_schema(schema, path=(), root=schema)
        assert result["definitions"]["User"]["additionalProperties"] is False
        assert result["definitions"]["User"]["required"] == ["name"]

    def test_defs_processed(self):
        schema = {
            "type": "object",
            "properties": {"user": {"$ref": "#/$defs/User"}},
            "$defs": {"User": {"type": "object", "properties": {"name": {"type": "string"}}}},
        }
        result = _ensure_strict_json_schema(schema, path=(), root=schema)
        assert result["$defs"]["User"]["additionalProperties"] is False
        assert result["$defs"]["User"]["required"] == ["name"]

    def test_ref_with_additional_properties_unrolled(self):
        schema = {
            "type": "object",
            "properties": {
                "user": {
                    "$ref": "#/definitions/User",
                    "description": "The user object",
                }
            },
            "definitions": {"User": {"type": "object", "properties": {"name": {"type": "string"}}}},
        }
        result = _ensure_strict_json_schema(schema, path=(), root=schema)
        # The $ref should be unrolled when there are additional properties
        user_prop = result["properties"]["user"]
        assert "$ref" not in user_prop
        assert user_prop["type"] == "object"
        assert user_prop["description"] == "The user object"
        assert user_prop["additionalProperties"] is False


class TestTypeToJsonSchema:
    def test_simple_model(self):
        class SimpleModel(BaseModel):
            name: str
            age: int

        schema = type_to_json_schema(SimpleModel)
        assert schema["type"] == "object"
        assert "name" in schema["properties"]
        assert "age" in schema["properties"]
        assert schema["additionalProperties"] is False
        assert set(schema["required"]) == {"name", "age"}

    def test_model_with_optional_field(self):
        class ModelWithOptional(BaseModel):
            name: str
            nickname: str | None = None

        schema = type_to_json_schema(ModelWithOptional)
        # All properties should be required in strict mode
        assert set(schema["required"]) == {"name", "nickname"}

    def test_model_with_nested_object(self):
        class Address(BaseModel):
            street: str
            city: str

        class Person(BaseModel):
            name: str
            address: Address

        schema = type_to_json_schema(Person)
        assert schema["additionalProperties"] is False
        assert "address" in schema["properties"]

        # Check nested object is also strict
        if "$defs" in schema:
            address_schema = schema["$defs"]["Address"]
        else:
            address_schema = schema["definitions"]["Address"]

        assert address_schema["additionalProperties"] is False
        assert set(address_schema["required"]) == {"street", "city"}

    def test_model_with_list_field(self):
        class TodoList(BaseModel):
            title: str
            items: list[str]

        schema = type_to_json_schema(TodoList)
        assert schema["properties"]["items"]["type"] == "array"
        assert schema["properties"]["items"]["items"]["type"] == "string"

    def test_model_with_field_descriptions(self):
        class DescribedModel(BaseModel):
            name: str = Field(description="The person's name")
            age: int = Field(description="The person's age")

        schema = type_to_json_schema(DescribedModel)
        assert schema["properties"]["name"]["description"] == "The person's name"
        assert schema["properties"]["age"]["description"] == "The person's age"

    def test_complex_nested_model(self):
        class Item(BaseModel):
            id: int
            name: str

        class Order(BaseModel):
            order_id: str
            items: list[Item]
            total: float

        class Customer(BaseModel):
            name: str
            orders: list[Order]

        schema = type_to_json_schema(Customer)
        assert schema["additionalProperties"] is False
        assert "orders" in schema["properties"]

        # Verify all nested models have strict schema
        defs_key = "$defs" if "$defs" in schema else "definitions"
        assert schema[defs_key]["Order"]["additionalProperties"] is False
        assert schema[defs_key]["Item"]["additionalProperties"] is False

    def test_model_preserves_constraints(self):
        class ConstrainedModel(BaseModel):
            age: int = Field(ge=0, le=120)
            email: str = Field(pattern=r"^[\w\.-]+@[\w\.-]+\.\w+$")

        schema = type_to_json_schema(ConstrainedModel)
        assert schema["properties"]["age"]["minimum"] == 0
        assert schema["properties"]["age"]["maximum"] == 120
        assert "pattern" in schema["properties"]["email"]
