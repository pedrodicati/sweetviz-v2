import configparser
import importlib.resources

config = configparser.ConfigParser()
# print("Config: " + os.path.abspath('sweetviz_defaults.ini'))
with importlib.resources.open_text("sweetviz", "sweetviz_defaults.ini") as f:
    config.read_file(f)
# config.read_file(open('sweetviz_defaults.ini'))
