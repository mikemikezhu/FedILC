import logging


class FedLogger:

    __instance = None
    __restart = None

    """
    Initialize
    """

    @staticmethod
    def getLogger(restart, filename):
        """ Static access method. """
        if FedLogger.__instance is None:
            FedLogger(filename)
        else:
            # Update logger
            FedLogger.__update_logger(filename)

        FedLogger.__restart = restart
        return FedLogger.__instance

    def __init__(self, filename):
        """ Virtually private constructor. """
        if FedLogger.__instance != None:
            raise Exception("Logger is a singleton!")
        else:
            FedLogger.__instance = self
            FedLogger.__update_logger(filename)

    """
    Private method
    """

    @staticmethod
    def __update_logger(filename):

        if FedLogger.__instance is None:
            raise Exception("Please init logger first!")

        logger = logging.getLogger()

        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        logger.setLevel(logging.INFO)

        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s")

        file_handler = logging.FileHandler(filename)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    """
    Public method
    """

    def log(self, msg):
        logger = logging.getLogger()
        logger.info("Restart - {}, {}".format(FedLogger.__restart, msg))
