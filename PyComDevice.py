import serial
import serial.tools.list_ports
import time
import threading
import logging
import PyLog
import atexit
import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt


class ComDevice(PyLog.LogClass):
    @classmethod
    def find_devices(cls) -> list:
        dev_list = serial.tools.list_ports.comports()
        logging.info("{}\t{}".format("Port", "Description"))
        for dev in dev_list:
            logging.info("{}\t{}".format(dev.device, dev.description))
        return dev_list

    # コンストラクタ/デストラクタ
    def __init__(self, port, dev_name: str = None, baudrate: int = 9600, bytesize=serial.EIGHTBITS, parity=serial.PARITY_NONE, stopbits=serial.STOPBITS_ONE, timeout: float = None, xonxoff: bool = False, rtscts: bool = False, write_timeout: float = None, dsrdtr: bool = False, inter_byte_timeout: float = None, fetch_interval: float = 0.5):
        # loggerの設定
        super().__init__(__file__)

        # 終了時に必ずCloseするようにする
        atexit.register(self.__del__)

        # pyserialの設定
        self.__dev = None
        self.__port = port
        self.__baudrate = baudrate
        self.__bytesize = bytesize
        self.__parity = parity
        self.__stopbits = stopbits
        self.__timeout = timeout
        self.__xonxoff = xonxoff
        self.__rtscts = rtscts
        self.__dsrdtr = dsrdtr
        self.__write_timeout = write_timeout
        self.__inter_byte_timeout = inter_byte_timeout

        # デバイスの設定
        self.__name = dev_name if dev_name is not None else port
        self.__DEBUG_MODE = False
        self.__interval = max(fetch_interval, 0.005)

        # デバイスの初期化
        self.__open_dev()
        self._init_dev()

        # データの設定
        self._cur_t, self._cur_val = time.perf_counter(), None
        self._clear_data()

        # threadの設定
        self.__thread = None
        self._lock = threading.Lock()
        self.__ctrl = 0

        self.log_info("init OK")
        self.__start_thread()

    def __del__(self):
        self.__stop_thread()
        self.__close_dev()

    # DeviceのOpen/Close/初期設定
    def __open_dev(self):
        try:
            self.log_info("open port {}".format(self.__port))
            self.__dev = serial.Serial(self.__port, self.__baudrate, self.__bytesize, self.__parity, self.__stopbits, self.__timeout, self.__xonxoff, self.__rtscts, self.__write_timeout, self.__dsrdtr, self.__inter_byte_timeout)
        except serial.SerialException:
            self.log_error("Failed to open {}".format(self.__port))
            self.set_debug_mode()
        except:
            self.log_exception("Failed to open {}".format(self.__port))
            self.set_debug_mode()

    def __close_dev(self):
        try:
            if self.__DEBUG_MODE:
                return
            if self.__dev is None:
                return
            if self.__dev.is_open:
                self.log_info('close port {}'.format(self.port))
                self.__dev.close()
        except:
            self.log_exception('failed to close dev')

    def _init_dev(self):
        if type(self) != ComDevice:
            self.log_warn('initialize dev [Not Implemented]')
        else:
            self.log_info('initialize dev')

    # DEBUG MODEへの変更
    def set_debug_mode(self):
        if self.__DEBUG_MODE:
            return

        self.log_warn("set to debug mode")
        self.__close_dev()

        self.__DEBUG_MODE = True
        self.__name += '*'
        self.__dev = None
        self.set_log_level(logging.DEBUG)

    # logの出力形式の変更
    def log(self, level: int, msg: str):
        level("{} {}".format(self.__name, msg))

    # deviceとの通信
    def _write(self, cmd: str):
        self.log_debug("<< {}".format(cmd))
        if not self.__DEBUG_MODE:
            try:
                self.__dev.write(cmd.encode())
            except:
                self.log_exception("Failed to write {}".format(cmd))

    def _read(self, size: int) -> str:
        ret = 'X' * size if self.__DEBUG_MODE else self.__dev.read(size).decode()
        self.log_debug(">> {}".format(ret))
        return ret

    def _read_line(self) -> str:
        ret = 'X' * 10 + '\n' if self.__DEBUG_MODE else self.__dev.readline().decode()
        self.log_debug(">> {}".format(ret))
        return ret

    def _read_until(self, expected: bytes, size: int = None) -> str:
        ret = 'X' * 10 + expected.decode() if self.__DEBUG_MODE else self.__dev.read_until(expected, size).decode()
        self.log_debug(">> {}".format(ret))
        return ret

    def _read_all(self) -> str:
        ret = 'X' * 10 + expected.decode() if self.__DEBUG_MODE else self.__dev.read_all().decode()
        self.log_debug(">> {}".format(ret))
        return ret

    # Theadの設定
    def __start_thread(self):
        if self.__thread is not None:
            return
        try:
            self.__thread = threading.Thread(target=self.__thread_main)
            self.__thread.setDaemon(True)
            self.__ctrl = 0
            self.log_info("Start thread")
            self.__thread.start()
        except:
            self.log_exception("Failed to start thread")

    def __stop_thread(self):
        self.__ctrl = -1
        if self.__thread is None:
            return
        if not self.__thread.is_alive():
            self.__thread = None
            return
        try:
            self.__thread.join(timeout=self.__interval*3)
            if self.__thread.is_alive():
                self.log_error('Thread join time out error')
            else:
                self.__thread = None
                self.log_info("Stop thread")
        except:
            self.log_exception("Failed to stop thread")

    def __thread_main(self):
        self.log_debug("Thread - Begin")
        while True:
            self._fetch_state()

            if self.__ctrl == -1:  # exit thread
                break
            elif self.__ctrl == 1:  # while capturing
                self._append_data()

            if self.__DEBUG_MODE:
                time.sleep(self.__interval)
        self.log_debug("Thread - End")

    # Getter
    def get_name(self): return self.__name

    # deviceの状態
    def _fetch_state(self):
        if type(self) != ComDevice:
            raise NotImplementedError('_fetch_state')
        self._lock.acquire()
        try:
            self.log_debug("<< fetch state")
            if self.__DEBUG_MODE:
                self._cur_t, self._cur_val = time.perf_counter(), random.random()
            else:
                self._cur_t, self._cur_val = time.perf_counter(), random.random()
        except:
            self.log_exception("Failed to fetch state")
            self.set_debug_mode()
        finally:
            self._lock.release()

    def is_debug_mode(self) -> bool:
        return self.__DEBUG_MODE

    def get_cur_val(self) -> (float, float):
        return self._cur_t.copy(), self._cur_val.copy()

    # deviceの状態の記録
    def start_cap_data(self):
        if self.__ctrl != 0:
            self.log_error('Failed to start_cap_data ctrl : {}'.format(self.__ctrl))
            return
        self._clear_data()
        self._append_data()
        self.__ctrl = 1
        self.log_debug("Start cap_data")

    def stop_cap_data(self):
        if self.__ctrl not in [1, 2]:
            self.log_error('Failed to stop_cap_data ctrl : {}'.format(self.__ctrl))
            return
        self._append_data()
        self.__ctrl = 0
        self.log_debug("Stop cap_data")

    def restart_cap_data(self):
        if self.__ctrl != 0 or len(self._data['Time']) == 0:
            self.log_error('Failed to restart_cap_data ctrl : {}'.format(self.__ctrl))
            return
        self._append_data()
        self.__ctrl = 1
        self.log_debug("Restart cap_data")

    def get_data(self, rename=False, max_num=-1) -> pd.DataFrame:
        if self.__ctrl > 0:
            self.log_error('Failed to get_data because capturing')
            return None
        df = pd.DataFrame.from_dict(self._data)
        if max_num > 0:
            step = max(int(df.shape[0]/max_num), 1)
            df = df[::step]

        if rename:
            for column in [col for col in df.columns if col != 'Time']:
                df.rename(columns={column: self.__name + '-' + column}, inplace=True)

        df.sort_values(by='Time', ascending=True, inplace=True)
        df.reset_index(inplace=True)

        return df

    def _append_data(self):
        self._data['Time'].append(self._cur_t)
        self._data['Value'].append(self._cur_val)

    def _clear_data(self):
        self._data = {'Time': [], 'Value': []}

    # deviceの状態が変わるまで待機
    def wait_for_val(self, val: float, dir: float, timeout: float):

        if dir == 0:
            self.log_error('Unexpected dir : {}'.format(val))

        if timeout < 0:
            self.log_warn('Wait for val : {} dir : {}'.format(val, dir))
        else:
            self.log_info('Wait for val : {} dir : {} timeout : {}'.format(val, dir, timeout))

        limit_time = time.perf_counter() + timeout
        while True:
            if dir > 0 and self._cur_val > val:
                return
            if dir < 0 and self._cur_val < val:
                return
            if timeout >= 0 and time.perf_counter() > limit_time:
                break
            time.sleep(self.__interval)
        self.log_error("Time out Error wait for val : {}, dir : {}".format(val, dir))

    def TEST(self, *, test_time: float = 10):
        if test_time < 1.0 or 600.0 < test_time:
            self.log_error("Unexpected test_time : {}".format(test_time))
            return
        self.log_info("==== TEST START ====")
        self.log_info("wait for {} s".format(test_time))
        self.start_cap_data()
        time.sleep(test_time)
        self.stop_cap_data()
        self.log_info("==== TEST END  ====")
        df = self.get_data(max_num=100)

        df['Time'] -= df['Time'].min()
        df['dTime'] = df['Time'].diff()/df['index'].diff()

        fps, fps_std = df['dTime'].mean(), df['dTime'].std()
        val, val_std = df['Value'].mean(), df['Value'].std()

        self.log_info("Period\t{:.3f} [σ={:.3f}] FPS:{:.2f}".format(fps, fps_std, 1/fps))
        self.log_info("Value\t{:.3f} [σ={:.3f}]".format(val, val_std))

        self.log_info("... Drawing ...")

        plt.rcParams["font.size"] = 14
        cmap = plt.get_cmap('tab20')
        fig, axes = plt.subplots(1, 2, tight_layout=True, figsize=(10, 4))

        # 時刻歴(Value)
        axes[0].plot(df['Time'], df['Value'], marker='o', color=cmap(0), zorder=1)
        # 平均値の描画
        x_data = np.array([df['Time'].min(), df['Time'].max()])
        y_data = np.array([val, val])
        axes[0].plot(x_data, y_data, color=cmap(2), ls='--', zorder=2)
        x_txt = x_data.mean()*0.3
        y_txt = val*1.1
        axes[0].text(x_txt, y_txt, "{:.3f} [σ={:.3f}]".format(val, val_std), color=cmap(2), zorder=3, backgroundcolor=(1, 1, 1, 0.8))
        # 軸ラベルの設定
        axes[0].set_xlabel('Time')
        axes[0].set_ylabel('Value')

        # 周期の描画
        axes[1].plot(df['index'], df['dTime'], color=cmap(0), zorder=1, marker='o')
        axes[1].plot([df['index'].min(), df['index'].max()], [fps, fps], color=cmap(2), ls='--', zorder=2)
        x_txt = (df['index'].max() - df['index'].min())*0.3
        y_txt = fps
        axes[1].text(x_txt, y_txt, "{:.3f} [σ={:.3f}]".format(fps, fps_std), color=cmap(2), zorder=3, backgroundcolor=(1, 1, 1, 0.8))
        axes[1].set_xlabel('Data no.')
        axes[1].set_ylabel('Period [s]')
        plt.show()
