from os import path
import configparser


def read_config(section, cfg_file='properties.ini'):
    config = configparser.ConfigParser()

    config.read(cfg_file)
    cfg = dict(config[section])

    # parse port as an integer
    if 'port' in cfg:
        cfg["port"] = config.getint(section, "port")

    return cfg
