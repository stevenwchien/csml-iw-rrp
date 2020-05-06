import sys
from datetime import datetime

class Logger(object):
    def __init__(self, fn):
        self.terminal = sys.stdout
        log_fn = 'logs/' + fn + datetime.now().strftime("%d-%m-%y-%H%M") + '.logfile'
        self.log = open(log_fn, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass
