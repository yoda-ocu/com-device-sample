from PyComDevice import ComDevice
import serial
import logging
import time
import random


class SampleDev(ComDevice):
    def __init__(self, port, dev_name: str = None, baudrate: int = 9600, bytesize=serial.EIGHTBITS, parity=serial.PARITY_NONE, stopbits=serial.STOPBITS_ONE, timeout: float = None, xonxoff: bool = False, rtscts: bool = False, write_timeout: float = None, dsrdtr: bool = False, inter_byte_timeout: float = None, fetch_interval: float = 0.5):
        super().__init__(port, dev_name, baudrate, bytesize, parity, stopbits, timeout, xonxoff, rtscts, write_timeout, dsrdtr, inter_byte_timeout, fetch_interval)

    def _fetch_state(self):
        self._lock.acquire()
        try:
            self.log_debug("<< fetch state")
            if self.is_debug_mode():
                self._cur_t, self._cur_val = time.perf_counter(), 10.0*random.random()
            else:
                self._cur_t, self._cur_val = time.perf_counter(), random.random()
        except:
            self.log_exception("Failed to fetch state")
            self.set_debug_mode()
        finally:
            self._lock.release()

    def zero(self):
        self._write("ZERO")

    def _init_dev(self):
        self.log_info('initialize dev')
        try:
            self.zero()
            self.log_info('initialize dev OK')
        except:
            self.log_exception('Failed to initialize')
            self.set_debug_mode()


if __name__ == "__main__":
    ComDevice.find_devices()
    a = SampleDev("PORT", fetch_interval=0)
    a.set_log_level(logging.INFO)
    a.TEST()
