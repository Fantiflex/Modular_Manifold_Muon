import inspect

import pytest


@pytest.fixture(autouse=True)
def show_test_description(request):
    """
    Print the description of every test before it runs.

    The description is taken from the test function's docstring.
    """
    description = inspect.getdoc(request.function)

    if description:
        print(f"\n\n[CHECK] {request.node.name}")
        for line in description.splitlines():
            print(f"        {line}")