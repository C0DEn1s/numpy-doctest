from refguide_check import Checker
import _pytest.doctest
import doctest


def pytest_addoption(parser):
    """Включает nice функцию с опцией --nice."""
    group = parser.getgroup('myplugin')
    group.addoption("--myplugin", action="store_true",
                    help="help help help")


def pytest_configure(config):

    def custom_checker() -> "doctest.OutputChecker":
        # return Checker(atol=0.00000001)
        return Checker(atol=0.000001)

    # if config.getoption('myplugin'):
    #     _pytest.doctest._get_checker = custom_checker
    doctest.OutputChecker = custom_checker()
