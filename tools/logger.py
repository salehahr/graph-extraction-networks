from typing import Dict, List, Union


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
    def __init__(self, filename: str, headers: List[str]):
        self._filename = filename

        self._delimiter = ","
        self._metric_headers = headers
        self._log_headers = ["img"] + headers + ["num_nodes", "time"]

        self._init_file()

    def _init_file(self):
        with open(self._filename, "w") as f:
            data_str = self._data_string(self._log_headers)
            f.write(data_str)

    def write(
        self, img_fp: str, data: Dict[str, Union[str, int]], num_nodes: int, time: float
    ):
        with open(self._filename, "a") as f:
            data_str = self._data_to_str(img_fp, data, num_nodes, time)
            f.write(data_str)

    def _data_to_str(
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

    def _data_string(self, l_strings: list) -> str:
        return self._delimiter.join(l_strings) + "\n"
