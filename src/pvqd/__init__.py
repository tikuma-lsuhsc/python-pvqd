"""Voice Foundation Pathological Voice Quality Database Reader module

TODO: download files directly from https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/9dz247gnyb-2.zip
"""


__version__ = "0.1.0.dev4"

import pandas as pd
from os import path
import numpy as np
from glob import glob as _glob
import re
import wave
from collections.abc import Sequence


class PVQD:
    def __init__(self, dbdir, default_type="/a/", padding=0.0, _timingpath=None):
        """PVQD constructor

        :param dbdir: path to the cdrom drive or the directory hosting a copy of the database
        :type dbdir: str
        """

        self.default_type = default_type
        self.default_padding = padding

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
    def task_types(self):
        return self._audio_timing.columns.get_level_values(0)

    def query(
        self,
        subset=None,
        include_cape_v=None,
        include_grbas=None,
        rating_stats=None,
        **filters,
    ):
        """query database

        :param subset: database columns to return, defaults to None
        :type subset: sequence of str, optional
        :param include_cape_v: True to include all CAPE-V scales, str or list of str to specify which scale, defaults to None
        :type include_cape_v: bool, 'breathiness', 'loudness','pitch','roughness','severity', list, optional
        :param include_grbas: True to include all GRBAS scales, str or list of str to specify which scale, defaults to None
        :type include_grbas: bool, 'asthenia', 'breathiness','grade','roughness','strain', list, optional
        :param rating_stats: Specify to return per-recording statistics across (up to) 4 raters, defaults to 'mean'
        :type rating_stats: operation supported by pandas aggregate function, optional
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
        if subset is not None:
            try:
                df = df[subset]
            except:
                ValueError(
                    f'At least one label in the "subset" argument is invalid: {subset}'
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
        type=None,
        auxdata_fields=None,
        include_cape_v=None,
        include_grbas=None,
        rating_stats=None,
        **filters,
    ):
        """get WAV filepaths, and starting and ending time markers

        :param type: utterance type, defaults to None, which is synonymous to "all"
        :type type: "all", "/a/", "/i/", "blue", "hard", "away", "egg", "lemon", or "peter", optional
        :param auxdata_fields: names of auxiliary data fields to return, defaults to None
        :type auxdata_fields: sequence of str, optional
        :param include_cape_v: True to include all CAPE-V scales, str or list of str to specify which scale, defaults to None
        :type include_cape_v: bool, 'breathiness', 'loudness','pitch','roughness','severity', list, optional
        :param include_grbas: True to include all GRBAS scales, str or list of str to specify which scale, defaults to None
        :type include_grbas: bool, 'asthenia', 'breathiness','grade','roughness','strain', list, optional
        :param rating_stats: Specify to return per-recording statistics across (up to) 4 raters, defaults to 'mean'
        :type rating_stats: operation supported by pandas aggregate function, optional
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

        if bool(type) and type != "all":
            try:
                df = df.join(self._audio_timing[type])
            except:
                raise ValueError(
                    f'Unknown type: {type} (must be one of "/a/", "/i/", "blue", "hard", "away", "egg", "lemon", or "peter")'
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
        type=None,
        auxdata_fields=None,
        normalize=True,
        include_cape_v=None,
        include_grbas=None,
        rating_stats=None,
        **filters,
    ):
        """iterate over data samples

        :param type: utterance type
        :type type: "rainbow" or "ah"
        :param channels: audio channels to read ('a', 'b', 0-1, or a sequence thereof),
                        defaults to None (all channels)
        :type channels: str, int, sequence, optional
        :param auxdata_fields: names of auxiliary data fields to return, defaults to None
        :type auxdata_fields: sequence of str, optional
        :param normalize: True to return normalized f64 data, False to return i16 data, defaults to True
        :type normalize: bool, optional
        :param diagnoses_filter: Function with the signature:
                                    diagnoses_filter(diagnoses: List[str]) -> bool
                                 Return true to include the database row to the query
        :type diagnoses_filter: Function
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



        Valid values of `auxdata_fields` argument
        ---------------------------------

        * All columns of the database specified in EXCEL50/TEXT/README.TXT Section 3.1
        (except for "DIAGNOSIS" and "#")
        * "DIAGNOSES" - A list containing all the original "DIAGNOSIS" associated with the subject
        * "NORM" - True if normal data, False if pathological data
        * "MDVP" - Short-hand notation to include all the MDVP parameter measurements: from "Fo" to "PER"

        Valid `filters` keyword arguments
        ---------------------------------

        * All columns of the database specified in EXCEL50/TEXT/README.TXT Section 3.1
        (except for "DIAGNOSIS" and "#")
        * "DIAGNOSES" - A list containing all the original "DIAGNOSIS" associated with the subject
        * "NORM" - True if normal data, False if pathological data

        Valid `filters` keyword argument values
        ---------------------------------------

        * A scalar value
        * For numeric and date columns, 2-element sequence to define a range: [start, end)
        * For all other columns, a sequence of allowable values
        """

        df = self.get_files(
            type,
            auxdata_fields,
            include_cape_v,
            include_grbas,
            rating_stats,
            **filters,
        )

        aux_cols = df.columns[3:]

        for id, file, tstart, tend, *auxdata in df.itertuples():
            framerate, x = self._read_file(file, tstart, tend, normalize)

            if bool(auxdata):
                yield id, framerate, x, pd.Series(
                    list(auxdata), index=aux_cols, name=id
                )
            else:
                yield id, framerate, x

    def read_data(self, id, type=None, normalize=True, padding=None):

        if not type:
            type = self.default_type

        if type != "all":
            tstart, tend = self._audio_timing.loc[id, type]

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
            type = ts[ts == tstart].index[0][0]
            i = np.where(ts.index.get_loc(type))[0]

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

    def __getitem__(self, key):
        return self.read_data(key)
