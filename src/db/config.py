import os
import configparser

CONFIG_FOLDER_PATH = os.path.dirname(__file__)


def read_config(section, cfg_file=CONFIG_FOLDER_PATH+"\properties.ini"):
    config = configparser.ConfigParser()

    config.read(cfg_file)
    cfg = dict(config[section])

    # parse port as an integer
    if 'port' in cfg:
        cfg["port"] = config.getint(section, "port")

    return cfg
