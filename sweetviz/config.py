import configparser
import importlib.resources


config = configparser.ConfigParser()
# print("Config: " + os.path.abspath('sweetviz_defaults.ini'))
try:
    # Use modern files() API instead of deprecated open_text
    files = importlib.resources.files("sweetviz")
    with (files / "sweetviz_defaults.ini").open() as f:
        config.read_file(f)
except AttributeError:
    # Fallback for older Python versions
    with importlib.resources.open_text("sweetviz", "sweetviz_defaults.ini") as f:
        config.read_file(f)
# config.read_file(open('sweetviz_defaults.ini'))
