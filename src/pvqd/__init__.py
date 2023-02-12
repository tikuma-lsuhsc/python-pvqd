"""Voice Foundation Pathological Voice Quality Database Reader module

TODO: download files directly from https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/9dz247gnyb-2.zip
"""


__version__ = "0.1.0.dev5"

import pandas as pd
from os import path
import numpy as np
from glob import glob as _glob
import re
import wave
from collections.abc import Sequence
from typing import Literal, Union, Callable, Tuple, Iterator, Optional

VoiceTask = Literal[
    "all", "/a/", "/i/", "blue", "hard", "away", "egg", "lemon", "peter"
]
CapeVScale = Literal["breathiness", "loudness", "pitch", "roughness", "severity"]
GRBASScale = Literal["asthenia", "breathiness", "grade", "roughness", "strain"]

DataField = Literal["Gender", "Age", "Diagnosis"]


class PVQD:
    def __init__(
        self,
        dbdir: str,
        default_task: VoiceTask = "/a/",
        default_padding: float = 0.0,
        _timingpath=None,
    ):
        """PVQD constructor

        :param dbdir: path to the cdrom drive or the directory hosting a copy of the database
        :type dbdir: str
        :param default_task: default voice task when task is not specified
        :type default_task: VoiceTask
        :param default_padding: default amount of extra samples to retrieve in seconds, defaults to 0.0
        :type default_padding: float
        """

        self.default_task = default_task
        self.default_padding = default_padding

        # database variables
        self._dir = None  # database dir
        self._df = None  # main database table
        self._df_rates = None  # patient diagnosis table
        self._df_times = None  # patient diagnoses series
        self._wavs = None

        # load the database
        self._load_db(dbdir, _timingpath)

    def _load_db(self, dbdir, timingpath):
        """load disordered voice database

        :param dbdir: path to the cdrom drive or the directory hosting a copy of the database
        :type dbdir: str

        * This function must be called at the beginning of each Python session
        * Database is loaded from the text file found at: <dbdir>/EXCEL50/TEXT/KAYCDALL.TXT
        * Only entries with NSP files are included
        * PAT_ID of the entries without PAT_ID field uses the "FILE VOWEL 'AH'" field value
        as the PAT_ID value

        """

        if self._dir == dbdir:
            return

        if timingpath is None:
            # use the default timing csv data
            timingpath = path.join(__path__[0], "timing.csv")

        xlsdir = path.join(dbdir, "Ratings Spreadsheets")

        df = pd.read_excel(
            path.join(xlsdir, "Demographics.xlsx"),
            dtype={"Age": "Int32", "Diagnosis ": "string"},
            nrows=297,
            keep_default_na=False,
            na_values={"Diagnosis": ""},
            # fmt:off
            converters={
                "Participant ID ": lambda v: v.strip().upper(),
                "Gender": lambda v: {"m": "male", "f": "female"}.get(v.lower(), v.lower()),
            },
            # fmt:on
        ).set_index("Participant ID ")
        df.columns = df.columns.str.strip()
        df.index.name = "ID"
        self._df = df

        df = pd.concat(
            {
                "CAPE-V": pd.read_excel(
                    path.join(xlsdir, "Ratings_both_scales.xlsx"),
                    sheet_name="CAPE-V",
                    usecols="A:B,G:N",
                    converters={
                        "File": lambda v: v.strip().upper(),
                        "Characteristics": lambda v: v.rsplit(" ", 1)[1],
                    },
                ).set_index(["File", "Characteristics"]),
                "GRBAS": pd.read_excel(
                    path.join(xlsdir, "Ratings_both_scales.xlsx"),
                    sheet_name="GRBAS",
                    usecols="A:B,G:N",
                    converters={
                        "File": lambda v: v.strip().upper(),
                        "Characteristics": lambda v: v.rsplit(" ", 1)[1],
                    },
                ).set_index(["File", "Characteristics"]),
            },
            names=["Scale", "ID", "Dimension"],
        )
        df.columns = pd.MultiIndex.from_tuples(
            [
                tuple(
                    int(v) for v in re.match(r"rater (\d) time (\d)", c, re.I).groups()
                )
                for c in df.columns
            ],
            names=["Rater", "Time"],
        )
        df = df.reorder_levels([0, 2, 1]).sort_index()
        self._df_rates = df

        df = pd.read_csv(
            timingpath,
            converters={"Participant ID": lambda v: v.strip().upper()},
        ).set_index(["Participant ID"])
        df.index.name = "ID"
        df.columns = pd.MultiIndex.from_tuples(
            [tuple(c.rsplit(" ", 1)) for c in df.columns], names=["Task", "Marker"]
        )
        self._audio_timing = df

        self._audio_dir = path.join(dbdir, "Audio Files")
        files = [path.basename(p) for p in _glob(path.join(self._audio_dir, "*.wav"))]
        self._audio_files = pd.Series(
            files,
            index=[re.match(r"(\D+\d+)", f)[1].upper() for f in files],
            name="File",
        )

    @property
    def voice_tasks(self) -> VoiceTask:
        return self._audio_timing.columns.get_level_values(0)

    def query(
        self,
        columns: Sequence[DataField] = None,
        include_cape_v: Union[bool, CapeVScale, Sequence[CapeVScale]] = None,
        include_grbas: Union[bool, GRBASScale, Sequence[GRBASScale]] = None,
        rating_stats: Union[str, Sequence[str], Callable] = None,
        **filters,
    ):
        """query database

        :param columns: database columns to return, defaults to None
        :type columns: sequence of str, optional
        :param include_cape_v: True to include all CAPE-V scales, str or list of str to specify which scale, defaults to None
        :type include_cape_v: Union[bool, CapeVScale, Sequence[CapeVScale]], optional
        :param include_grbas: True to include all GRBAS scales, str or list of str to specify which scale, defaults to None
        :type include_grbas: Union[bool, GRBASScale, Sequence[GRBASScale]], optional
        :param rating_stats: Specify to return per-recording statistics across (up to) 4 raters.
                             any operation supported by pandas aggregate function, defaults to ['mean','min','max']
        :type rating_stats: Union[str, Sequence[str], Callable], optional
        :param **filters: query conditions (values) for specific per-database columns (keys)
        :type **filters: dict
        :return: query result
        :rtype: pandas.DataFrame

        Valid `filters` keyword argument values
        ---------------------------------------

        * A scalar value
        * For numeric and date columns, 2-element sequence to define a range: [start, end)
        * For all other columns, a sequence of allowable values

        """

        # work on a copy of the dataframe
        df = self._df.copy(deep=True)

        # apply the filters to reduce the rows
        for fcol, fcond in filters.items():
            try:
                s = df[fcol]
            except:
                raise ValueError(f"{fcol} is not a valid column label")

            try:  # try range/multi-choices
                if s.dtype.kind in "iufcM":  # numeric/date
                    # 2-element range condition
                    df = df[(s >= fcond[0]) & (s < fcond[1])]
                else:  # non-numeric
                    df = df[s.isin(fcond)]  # choice condition
            except:
                # look for the exact match
                df = df[s == fcond]

        # return only the selected columns
        if columns is not None:
            try:
                df = df[columns]
            except:
                ValueError(
                    f'At least one label in the "columns" argument is invalid: {columns}'
                )

        def prep_rating_stats(scale, dfr, includes, stats):
            dims = []
            if isinstance(includes, str):
                dims.append(includes.capitalize())
            elif isinstance(includes, Sequence):
                dims = list(map(str.capitalize, includes))
            if len(dims):
                dfr = dfr.loc[dims]
            dfr.columns = dfr.columns.get_level_values(0)
            dfr = dfr.agg(stats, axis=1).unstack(0)
            dfr.columns = [f"{scale} {c[1]} {c[0]}" for c in dfr.columns]
            return dfr

        if not bool(rating_stats):
            rating_stats = ["mean", "std", "min", "max"]
        elif isinstance(rating_stats, str):
            rating_stats = [rating_stats]

        if bool(include_cape_v):
            df = df.join(
                prep_rating_stats(
                    "CAPE-V", self._df_rates.loc["CAPE-V"], include_cape_v, rating_stats
                )
            )

        if bool(include_grbas):
            df = df.join(
                prep_rating_stats(
                    "GRBAS", self._df_rates.loc["GRBAS"], include_grbas, rating_stats
                )
            )

        return df

    def get_files(
        self,
        task: VoiceTask = None,
        auxdata_fields: Sequence[DataField] = None,
        include_cape_v: Union[bool, CapeVScale, Sequence[CapeVScale]] = None,
        include_grbas: Union[bool, GRBASScale, Sequence[GRBASScale]] = None,
        rating_stats: Union[str, Sequence[str], Callable] = None,
        **filters,
    ):
        """get WAV filepaths, and starting and ending time markers

        :param task: utterance task, defaults to None, which is synonymous to "all"
        :type task: VoiceTask, optional
        :param auxdata_fields: names of auxiliary data fields to return, defaults to None
        :type auxdata_fields: Sequence[DataField], optional
        :param include_cape_v: True to include all CAPE-V scales, str or list of str to specify which scale, defaults to None
        :type include_cape_v: Union[bool, CapeVScale, Sequence[CapeVScale]], optional
        :param include_grbas: True to include all GRBAS scales, str or list of str to specify which scale, defaults to None
        :type include_grbas: Union[bool, GRBASScale, Sequence[GRBASScale]], optional
        :param rating_stats: Specify to return per-recording statistics across (up to) 4 raters.
                             any operation supported by pandas aggregate function, defaults to ['mean','min','max']
        :type rating_stats: Union[str, Sequence[str], Callable], optional
        :param **filters: query conditions (values) for specific per-database columns (keys)
        :type **filters: dict
        :return: data frame containing file path, start and end time marks, and auxdata
        :rtype: pandas.DataFrame

        Valid values of `auxdata_fields` argument
        ---------------------------------

        * All columns of the database specified in EXCEL50/TEXT/README.TXT Section 3.1
        (except for "DIAGNOSIS" and "#")

        Valid `filters` keyword arguments
        ---------------------------------

        * All columns of the database specified in EXCEL50/TEXT/README.TXT Section 3.1
        (except for "DIAGNOSIS" and "#")

        Valid `filters` keyword argument values
        ---------------------------------------

        * A scalar value
        * For numeric and date columns, 2-element sequence to define a range: [start, end)
        * For all other columns, a sequence of allowable values
        """

        dir = self._audio_dir
        df = pd.DataFrame(self._audio_files.map(lambda v: path.join(dir, v)))

        if bool(task) and task != "all":
            try:
                df = df.join(self._audio_timing[task])
            except:
                raise ValueError(
                    f'Unknown task: {task} (must be one of "/a/", "/i/", "blue", "hard", "away", "egg", "lemon", or "peter")'
                )

        # eliminate entries without data
        df = df[df[["File", "Start", "End"]].notna().all(axis=1)]

        # get
        if (
            len(filters)
            or bool(auxdata_fields)
            or bool(include_cape_v)
            or bool(include_grbas)
        ):
            auxdata = self.query(
                auxdata_fields,
                include_cape_v,
                include_grbas,
                rating_stats,
                **filters,
            )

            if not bool(auxdata_fields):
                auxdata.drop(["Gender", "Age", "Diagnosis"], axis=1, inplace=True)

            df = df.join(auxdata, how="inner")

        return df

    def iter_data(
        self,
        task: VoiceTask = None,
        auxdata_fields: Sequence[DataField] = None,
        normalize: bool = True,
        include_cape_v: Union[bool, CapeVScale, Sequence[CapeVScale]] = None,
        include_grbas: Union[bool, GRBASScale, Sequence[GRBASScale]] = None,
        rating_stats: Union[str, Sequence[str], Callable] = None,
        **filters,
    ) -> Iterator[Tuple[int, np.array, Optional[pd.Series]]]:
        """iterate over data samples

        :param task: utterance task
        :type task: VoiceTask
        :param auxdata_fields: names of auxiliary data fields to return, defaults to None
        :type auxdata_fields: sequence of str, optional
        :param normalize: True to return normalized f64 data, False to return i16 data, defaults to True
        :type normalize: bool, optional
        :param include_cape_v: True to include all CAPE-V scales, str or list of str to specify which scale, defaults to None
        :type include_cape_v: Union[bool, CapeVScale, Sequence[CapeVScale]], optional
        :param include_grbas: True to include all GRBAS scales, str or list of str to specify which scale, defaults to None
        :type include_grbas: Union[bool, GRBASScale, Sequence[GRBASScale]], optional
        :param rating_stats: Specify to return per-recording statistics across (up to) 4 raters.
                             any operation supported by pandas aggregate function, defaults to ['mean','min','max']
        :type rating_stats: Union[str, Sequence[str], Callable], optional
        :param **filters: query conditions (values) for specific per-database columns (keys)
        :type **filters: dict
        :yield:
            - sampling rate : audio sampling rate in samples/second
            - data  : audio data, 1-D for 1-channel NSP (only A channel), or 2-D of shape
                    (Nsamples, 2) for 2-channel NSP
            - auxdata : (optional) requested auxdata of the data if auxdata_fields is specified
        :ytype: tuple(int, numpy.ndarray(int16)[, pandas.Series])

        Iterates over all the DataFrame columns, returning a tuple with the column name and the content as a Series.

        Yields

            labelobject

                The column names for the DataFrame being iterated over.
            contentSeries

                The column entries belonging to each label, as a Series.
        """

        df = self.get_files(
            task,
            auxdata_fields,
            include_cape_v,
            include_grbas,
            rating_stats,
            **filters,
        )
        """
                :param include_cape_v: True to include all CAPE-V scales, str or list of str to specify which scale, defaults to None
                :type include_cape_v: Union[bool, CapeVScale, Sequence[CapeVScale]], optional
                :param include_grbas: True to include all GRBAS scales, str or list of str to specify which scale, defaults to None
                :type include_grbas: Union[bool, GRBASScale, Sequence[GRBASScale]], optional
                :param rating_stats: Specify to return per-recording statistics across (up to) 4 raters.
                                    any operation supported by pandas aggregate function, defaults to ['mean','min','max']
                :type rating_stats: Union[str, Sequence[str], Callable], optional
        """
        aux_cols = df.columns[3:]

        for id, file, tstart, tend, *auxdata in df.itertuples():
            framerate, x = self._read_file(file, tstart, tend, normalize)

            if bool(auxdata):
                yield id, framerate, x, pd.Series(
                    list(auxdata), index=aux_cols, name=id
                )
            else:
                yield id, framerate, x

    def read_data(
        self,
        id: str,
        task: VoiceTask = None,
        normalize: bool = True,
        padding: float = None,
    ) -> Tuple[int, np.array]:
        """read data samples

        :param id: recording ID
        :type id: str
        :param task: utterance task, defaults to None (use default_task)
        :type task: VoiceTask
        :param normalize: True to return normalized f64 data, False to return i16 data, defaults to True
        :type normalize: bool, optional
        :param padding: default amount of extra samples to retrieve in seconds, defaults to None (default_padding)
        :type padding: float
        :return: sampling rate in Samples/second and numpy array
        :rtype: Tuple[int, np.array]
        """
        if not task:
            task = self.default_task

        if task != "all":
            tstart, tend = self._audio_timing.loc[id, task]
        else:
            tstart = tend = None

        return self._read_file(
            path.join(self._audio_dir, self._audio_files[id]),
            tstart,
            tend,
            normalize,
            padding,
        )

    def _read_file(self, file, tstart=None, tend=None, normalize=True, padding=None):
        if not padding:
            padding = self.default_padding

        if padding:
            id = re.match(r"(\D+\d+)", path.basename(file))[1].upper()

            ts = self._audio_timing.loc[id].sort_values()
            task = ts[ts == tstart].index[0][0]
            i = np.where(ts.index.get_loc(task))[0]

            tstart -= padding
            tend += padding

            talt = ts.iloc[i[0] - 1] if i[0] > 0 else 0.0
            if tstart < talt:
                tstart = talt

            if i[1] + 1 < len(ts):
                talt = ts.iloc[i[1] + 1]
                if tend > talt:
                    tend = talt

        with wave.open(file) as wobj:
            nchannels, sampwidth, framerate, nframes, *_ = wobj.getparams()
            dtype = f"<i{sampwidth}" if sampwidth > 1 else "u1"
            n0 = round(framerate * tstart) if tstart else 0
            n1 = round(framerate * tend) if tend else nframes
            b = wobj.readframes(n1)

        x = np.frombuffer(b, dtype, offset=n0 * nchannels * sampwidth)

        if nchannels > 1:
            x = x.reshape(-1, nchannels)

        if normalize:
            x = x / 2.0 ** (sampwidth * 8 - 1 if sampwidth > 1 else 8)

        return framerate, x

    def __getitem__(self, key) -> Tuple[int, np.array]:
        """get recording data by id

        :param key: recording id
        :type key: str
        :return: sampling rate in Samples/second and numpy array
        :rtype: Tuple[int, np.array]

        The function returns acoustic data of the default voice task 
        (PVQD.default_task) with default padding (PVQD.default_padding)
        """
        return self.read_data(key)
