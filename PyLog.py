import logging
logging.basicConfig(level=logging.INFO)


class LogClass():
    def __init__(self, name):
        self.__logger = logging.getLogger(name)
        self.__logger.info("import OK")

    def log(self, level, msg: str):
        level("{}".format(msg))

    def log_debug(self, msg: str):
        self.log(self.__logger.debug, msg)

    def log_info(self, msg: str):
        self.log(self.__logger.info, msg)

    def log_warn(self, msg: str):
        self.log(self.__logger.warning, msg)

    def log_error(self, msg: str):
        self.log(self.__logger.error, msg)

    def log_exception(self, msg: str):
        self.log(self.__logger.exception, msg)

    def set_log_level(self, level):
        try:
            self.__logger.setLevel(level)
            self.log_warn("Set log_level:{}".format(logging._levelToName[level]))
        except:
            self.log_exception("Error at set level")

    def get_log_level(self):
        return self.__logger.level
