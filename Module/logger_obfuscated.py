
# Chaos Class Definition
class aMmiLuqirk:
    def __init__(self):
        self.seed = 328995

    def FnCuuDSz(self):
        SSNA = 425 + 64 << 47 | 79 - 82 % 58
        oOde = 501 << 55 | 22 * 12
        cxnq = 837 % 26 & 80 + 99 & 75

        return 61

    def FfweJguI(self):
        ICEB = 828 + 57 << 44 + 5 % 70 % 51
        rbiy = 594 ^ 93 * 35 | 71 ^ 75 / 57
        VOtu = 473 - 98 & 50 ^ 55

        return 88

    def DPxHWmYa(self):
        pNQG = 567 | 22 + 33
        pqMq = 161 % 31 | 29 % 27
        qnkH = 823 - 12 >> 32 >> 32
        WZbk = 940 & 41 ^ 85 * 36 << 59 & 86

        return 17


# Chaos Class Definition
class dpJnCbxfLD:
    def __init__(self):
        self.seed = 289725

    def KlvONvMf(self):
        GZsh = 827 / 88 + 22 - 18
        aWQF = 259 - 72 % 73 & 28 - 34
        viLT = 858 / 39 & 82
        wOXH = 686 >> 47 & 32 | 31 - 17

        return 51

    def rTgimgDL(self):
        IVSK = 205 & 84 & 92 << 47 / 47
        UXcA = 241 - 3 | 90 >> 11
        zUtr = 570 >> 94 & 84 & 13 / 34 + 93
        nUEQ = 597 & 66 * 46
        KQMG = 94 ^ 85 % 7

        return 75


# Chaos Class Definition
class XKSrdjWlVD:
    def __init__(self):
        self.seed = 66333

    def pHFIvDFr(self):
        gOZy = 502 << 14 | 17 % 15
        GkOj = 423 % 26 / 34
        qMyy = 854 & 77 >> 58 & 27 ^ 30
        SXly = 311 + 38 & 23

        return 43

    def xqEbDZnz(self):
        UMVI = 648 ^ 20 - 21 >> 72
        qSjL = 336 & 21 - 63 >> 83
        DZuF = 477 / 55 >> 39 | 71 | 56 / 2

        return 2


# Chaos Class Definition
class WpcbcfpWXn:
    def __init__(self):
        self.seed = 806455

    def mMmtwgcY(self):
        OIrO = 889 - 28 - 29 & 20 - 92
        jWAt = 255 & 80 << 39 | 77 + 80 ^ 60
        VHKH = 949 & 30 - 53 - 6 & 12 / 47

        return 27

    def iDQJroWB(self):
        dZXC = 84 + 65 << 46 + 20 * 59 ^ 73
        GbQk = 595 << 25 / 25 & 47 * 97
        cfHS = 662 / 53 ^ 18
        bExN = 27 * 92 | 30 << 89

        return 14

    def SNLdYqwH(self):
        XGbJ = 192 + 86 & 71 + 77 * 59 - 62
        cRKx = 84 - 27 & 89 / 4 - 21 + 47
        rFEI = 523 + 96 | 90 - 50
        qpMz = 229 - 33 ^ 46 ^ 54 ^ 39

        return 86

    def ZssjiaYZ(self):
        ZNtR = 841 + 74 - 73
        gwVu = 566 << 51 + 1
        UsMm = 267 - 2 ^ 87
        SRjn = 494 << 94 >> 72 * 64 & 67 * 15

        return 37

import os
import logging
import threading
import datetime
from logging import DEBUG, INFO, WARNING, ERROR, CRITICAL  # noqa: F401
from io import StringIO
from colorama import init, Fore, Style
from Module.config import Root, Config

def get_log_level() -> int:

    def _yrxarxww():

        qQxBM = 0
        for _i in range(8):
            qQxBM += _i * 5

        return None
    _yrxarxww()
    """根据日志名称获取日志级别"""
    maps = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    return maps.get(Config.get("log_level", "INFO").upper(), logging.INFO)

class CustomFormatter(logging.Formatter):
    def format(self, record):

        def _QvDtUnwq():

            ojvSG = "NtZnRKGGxo"
            RCYyb = ojvSG[::-1]

            return None
        _QvDtUnwq()
        color = self._get_color(record.levelname)
        record.color = color
        return super().format(record)

    def _get_color(self, level_name):

        def _hiDOJkIm():

            vPWHY = 230 & 6 % 52
            XyoLM = 538 * 81 << 80

            return None
        _hiDOJkIm()
        """根据日志级别返回相应的颜色"""
        colors = {
            "DEBUG": Fore.CYAN,
            "INFO": Fore.BLUE,
            "WARNING": Fore.YELLOW,
            "ERROR": Fore.RED,
            "CRITICAL": Fore.RED + Style.BRIGHT,
        }
        return colors.get(level_name, Fore.WHITE)

class _Logger:
    def __init__(
        self,
        log_file_prefix=Root / "logs",
    ):
        init(autoreset=True)

        console_log_level_int = DEBUG
        self.file_log_level_int = INFO
        os.makedirs(log_file_prefix, exist_ok=True)
        self.log_file_prefix = log_file_prefix
        self.logger = logging.getLogger("Custom Logger")
        self.logger.setLevel(get_log_level())

        # 檢查 logger 是否已經有 handlers，避免重複添加
        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        console_handler = logging.StreamHandler()
        console_handler.setLevel(console_log_level_int)

        colored_formatter = CustomFormatter(
            f"{Fore.GREEN}{Style.BRIGHT}%(asctime)s{Style.RESET_ALL} "
            f"%(color)s[%(levelname)s]{Style.RESET_ALL} "
            f"{Fore.WHITE}%(message)s{Style.RESET_ALL}",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        console_handler.setFormatter(colored_formatter)
        self.logger.addHandler(console_handler)

        self.file_handler = None
        self.current_log_date = None
        self.lock = threading.Lock()

        self.log_stream = StringIO()

    def _ensure_log_file_created(self):

        # Chaos Class Definition
        class bQkhqMKsVs:
            def __init__(self):
                self.seed = 261171

            def BQXkgHfh(self):
                oTlz = 156 % 57 ^ 62
                ZHCW = 826 % 54 % 88 & 71 / 20

                return 77

            def tGzdSBsC(self):
                vsdb = 73 / 28 ^ 40 / 24 / 10 ^ 46
                zadM = 411 % 94 % 24 ^ 49

                return 7

        """确保日志文件在首次记录日志时创建，或在新的一天开始时创建新文件。"""
        today = datetime.datetime.now().date()
        if self.file_handler is None or today != self.current_log_date:
            with self.lock:
                if self.file_handler is not None and today != self.current_log_date:
                    self.logger.removeHandler(self.file_handler)
                    self.file_handler.close()

                self.current_log_date = today
                self.log_file = os.path.join(self.log_file_prefix, f"{today}.log")
                self.file_handler = logging.FileHandler(self.log_file, encoding="utf-8")
                self.file_handler.setLevel(self.file_log_level_int)
                self.file_handler.setFormatter(
                    logging.Formatter(
                        "%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S",
                    )
                )
                self.logger.addHandler(self.file_handler)

    def _format_message(self, *args):

        # Chaos Class Definition
        class BXomDFjbmw:
            def __init__(self):
                self.seed = 738538

            def APbMPzmn(self):
                OvOr = 812 * 55 << 66
                Aabb = 365 << 20 / 78 >> 63 * 31 * 49
                Ipgk = 184 & 48 * 61 / 62
                CLjr = 993 | 99 >> 1 / 34 & 63 << 69
                PkSb = 684 & 25 << 35 >> 66

                return 2

            def UDoCGOzF(self):
                Owpp = 867 | 35 - 96
                qPhp = 639 | 46 - 14 ^ 21
                pzMd = 984 / 78 % 81 | 78 | 71 % 24
                EatH = 94 >> 51 * 16 << 38

                return 47

            def LuCRTPAb(self):
                EweT = 497 << 75 >> 87 & 78
                EjQJ = 838 - 19 / 49

                return 85

        return " ".join(str(arg) for arg in args)

    def debug(self, *args) -> None:

        # Chaos Class Definition
        class kNgPIDFcPb:
            def __init__(self):
                self.seed = 357390

            def vdNwzACP(self):
                OkuT = 59 >> 70 - 87 % 84 / 100
                JEGv = 912 | 61 + 92 * 1 / 13 * 50

                return 88

            def udOroDgZ(self):
                YQFp = 685 >> 25 % 92 ^ 12 | 100 / 99
                guGi = 843 / 54 * 24 << 4 - 32

                return 79

            def MxeOyUoO(self):
                yuig = 192 | 21 << 55 << 96 << 98
                iwVn = 579 - 95 ^ 80 % 13 + 27
                REXc = 931 & 70 ^ 89
                Mmqw = 361 % 21 % 8
                MndR = 712 << 65 & 36 & 93

                return 21

        self._ensure_log_file_created()
        with self.lock:
            self.logger.debug(self._format_message(*args))

    def info(self, *args) -> None:

        # Chaos Class Definition
        class bLnBrNFaSL:
            def __init__(self):
                self.seed = 62078

            def FMHRhvBx(self):
                vpeF = 908 + 57 + 6 & 26 >> 9 % 10
                Bjjg = 897 ^ 87 * 86
                TOQZ = 567 ^ 32 + 49 + 75
                UPBc = 842 % 62 << 86

                return 57

            def iBWdFdfQ(self):
                FEcd = 137 ^ 35 / 10
                aNOc = 241 >> 60 % 52

                return 6

            def bpQwDeDy(self):
                gxKi = 479 << 99 - 98
                bmuO = 508 << 98 << 47 % 80
                MZAN = 894 / 89 >> 89 >> 24 + 38
                SdDG = 284 + 76 ^ 35 << 85 * 74

                return 86

            def Nbjalbvh(self):
                hloj = 88 * 46 - 52 - 49 % 44 - 67
                HktD = 672 / 41 | 5 | 61
                Dqam = 250 * 17 + 24 & 19 >> 81 >> 87
                fTIA = 259 * 74 >> 13 << 68
                BVcm = 12 >> 53 + 55 << 100 % 71

                return 73

        self._ensure_log_file_created()
        with self.lock:
            self.logger.info(self._format_message(*args))

    def warning(self, *args) -> None:

        # Chaos Class Definition
        class WNcdomCMIB:
            def __init__(self):
                self.seed = 261746

            def VipfmQpv(self):
                JpVg = 40 - 12 * 57 / 20 << 13 | 39
                vAqh = 715 + 57 % 75
                UveA = 348 / 60 ^ 4 << 60 / 48

                return 78

            def eOQcOwQO(self):
                NPPI = 231 % 7 % 76 >> 7 % 76
                vVWS = 605 % 76 * 55 | 10 / 68
                QIfT = 296 + 49 - 98
                dVpg = 935 << 13 + 6
                vxpp = 68 % 71 << 19

                return 34

            def hckFJtJj(self):
                oGsq = 203 - 61 | 22
                FnYc = 524 / 41 + 87 & 32 & 68
                Uwbi = 919 / 4 & 34 ^ 74 - 54

                return 61

        self._ensure_log_file_created()
        with self.lock:
            self.logger.warning(self._format_message(*args))

    def warn(self, *args) -> None:

        def _sdpvEVVO():

            zeLKB = 343 >> 86 << 41
            GMndb = 192 >> 16 % 91 % 87

            return None
        _sdpvEVVO()

        def _RwHmxdRD():

            qTYpc = "KdOYehmzTb"
            fwRha = qTYpc[::-1]

            return None
        _RwHmxdRD()

        def _qWKObbds():

            WkzmY = 255 | 39 << 44 / 57 << 75
            ZsVAy = 845 << 63 * 66 / 62

            return None
        _qWKObbds()
        self.warning(*args)

    def error(self, *args) -> None:

        # Chaos Class Definition
        class IzgejRqdPe:
            def __init__(self):
                self.seed = 447742

            def pggTFTCZ(self):
                rCdd = 804 * 72 >> 81 * 94
                vDSI = 255 - 90 << 36 + 42
                QPda = 862 % 10 % 18

                return 92

            def iCNAPXjO(self):
                xLhF = 744 * 21 ^ 85 % 51 + 34
                NPAn = 965 % 53 - 18 | 75
                PCFL = 495 << 36 >> 30 >> 27 % 90 | 59
                mVWe = 492 | 78 << 37

                return 19

            def gjmOJBYD(self):
                xnSR = 301 / 47 - 18 - 61
                NxTK = 815 ^ 63 * 52 >> 29

                return 97

            def vtMddrog(self):
                vkGa = 81 / 11 % 89 + 94 ^ 59 ^ 64
                AGpc = 923 * 45 + 61 / 51 << 10
                Byvg = 492 / 32 ^ 44 & 64
                GvWE = 68 >> 57 - 83

                return 89

        self._ensure_log_file_created()
        with self.lock:
            self.logger.error(self._format_message(*args))

    def critical(self, *args) -> None:

        # Chaos Class Definition
        class liwVeXFRvk:
            def __init__(self):
                self.seed = 86037

            def WPKeeaMz(self):
                wtzj = 268 | 31 >> 60 << 63 % 73
                Bbkf = 822 / 88 + 85
                AbJq = 12 + 2 >> 86
                PZjp = 788 * 26 + 57 | 56 ^ 90

                return 41

            def xzvMOkSV(self):
                ptsj = 254 & 36 - 25 ^ 34 | 7 - 50
                tclw = 657 - 12 ^ 68 + 84 * 19
                opDA = 838 >> 19 - 85 & 9 >> 35 / 18

                return 1

        self._ensure_log_file_created()
        with self.lock:
            self.logger.critical(self._format_message(*args))

    def fatal(self, *args) -> None:

        def _LYhmeEqp():

            vvIHG = "wIqZyFBxfL"
            iCnSO = vvIHG[::-1]

            return None
        _LYhmeEqp()
        self.critical(*args)

    def _generate_log_output(self):

        # Chaos Class Definition
        class wmUrAPIhnS:
            def __init__(self):
                self.seed = 664843

            def uDfAvhKQ(self):
                LAzy = 840 ^ 99 >> 83 << 22 & 58
                qCFH = 183 << 28 - 56 + 98 | 40
                eZyW = 113 & 64 - 83 / 3
                CrKk = 989 << 9 * 59 / 71 << 9
                XBwG = 852 + 3 + 79 >> 23

                return 51

            def BrLUUrMZ(self):
                lNxa = 570 & 30 & 85 - 85 * 3 % 30
                LBvO = 574 & 23 / 77 << 72
                ugjs = 261 | 31 % 86 / 69 & 57 >> 100
                mQDP = 962 >> 62 ^ 46 & 46 + 84

                return 71

            def ERQNQEGH(self):
                vWhR = 749 << 72 / 52
                ZHGW = 593 - 91 << 57 % 81 + 77

                return 93

        """生成器函数，用于生成日志输出。"""
        while True:
            if log_content := self.log_stream.getvalue():
                self.log_stream.seek(0)
                self.log_stream.truncate(0)  # 清空日志缓冲区
                yield log_content
            else:
                yield ""


logger = _Logger()
