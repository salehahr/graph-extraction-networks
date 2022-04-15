from typing import Dict, List, Optional, Union

from tools.NetworkType import NetworkType


class Metrics:
    def __init__(self):
        super(Metrics, self).__init__()

    def append(self, params):
        new_attributes = dict(params)
        self.__dict__.update(new_attributes)

    def get(self, key):
        return self.__dict__.get(key)

    def set(self, key, val):
        self.__dict__[key] = val

    def __iter__(self):
        """Allow object to be iterable"""
        for attr, value in self.__dict__.items():
            yield attr, value


class Logger:
    def __init__(self, filename: str, headers: List[str], network: NetworkType):
        self._filename = filename

        self._delimiter = ","
        self._metric_headers = headers

        if network == NetworkType.ADJ_MATR_NN:
            self._log_headers = ["img"] + headers + ["num_nodes", "time"]
        else:
            self._log_headers = ["batch"] + headers

        self._network = network

        self._init_file()

    def _init_file(self):
        with open(self._filename, "w") as f:
            data_str = self._data_string(self._log_headers)
            f.write(data_str)

    def write(
        self,
        data: Dict[str, Union[str, int]],
        batch: Optional[int] = None,
        img_fp: Optional[str] = None,
        num_nodes: Optional[int] = None,
        time: Optional[float] = None,
    ):
        with open(self._filename, "a") as f:
            if self._network == NetworkType.ADJ_MATR_NN:
                data_str = self._data_to_str_adj_matr(img_fp, data, num_nodes, time)
            elif self._network == NetworkType.EDGE_NN:
                data_str = self._data_to_str_edge(batch, data)
            f.write(data_str)

    def _data_to_str_adj_matr(
        self,
        img_fp: str,
        data: Dict[str, Union[str, float]],
        num_nodes: int,
        time: float,
    ) -> str:
        l_data = (
            [img_fp]
            + [
                f"{data[h]:.5f}" if isinstance(data[h], float) else str(data[h])
                for h in self._metric_headers
            ]
            + [str(num_nodes), f"{time:.5f}"]
        )
        return self._data_string(l_data)

    def _data_to_str_edge(self, batch: int, data: Dict[str, float]) -> str:
        data["f1"] = calculate_f1(data["precision"], data["recall"])
        l_data = (
            [str(batch)]
            + [str(int(data[h])) for h in self._metric_headers[:4]]
            + [f"{data[h]:.5f}" for h in self._metric_headers[4:]]
        )
        return self._data_string(l_data)

    def _data_string(self, l_strings: list) -> str:
        return self._delimiter.join(l_strings) + "\n"


def calculate_f1(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall)
