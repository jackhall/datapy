"""
This alternative dataframe comes from much frustration with pandas, especially
its interface. Many of its problems are ultimately historical; it was not quite
consistent in many ways to start out, and attempts to make it consistent
without breaking client code generally made it more complicated. Other problems
come from its focus on imperative manipulation rather than a functional
abstraction.

That said, this alternative owes much to pandas. The concept of hierarchical
indexes and how they can be used to create multi-dimensionality, all the
struggles over handling and representing null values, an understanding of
what users need from a dataframe... datapy would not be possible without this
collective experience.

I will focus on the Zen of Python, in particular:
- Simple is better than complex.
- There should be one-- and preferably only one --obvious way to do it.
- If the implementation is hard to explain, it's a bad idea.
- Namespaces are one honking great idea -- lets do more of those!

A dataframe should not surprise the user!

If this interface doesn't do everything you want, extend it with a mixin!
It's a great way to get more advanced functionality for various use cases
while keeping those use cases in separate namespaces. Just keep to the
public interface, ok? :)
"""
import typing as ty

import attr
import numpy as np
import pandas as pd


class Index:
    def find(self, idx):  # return whatever is needed to index a numpy array
        ...

    @property
    def shape(self):
        ...

    @property
    def is_sparse(self):
        ...

    def __contains__(self, idx):
        ...

    def __iter__(self):
        ...

    def __len__(self):
        ...


T = ty.TypeVar('T')


@attr.s(auto_attribs=True)
class Field(ty.Generic[T]):
    """ sequence of T """
    _array: np.ndarray = attr.ib()
    _null_mask: np.ndarray = attr.ib()
    _index: pd.MultiIndex  # optional?

    def __getitem__(self, idx) -> ty.Optional[T]:  # idx could be a slice
        ...

    def __setitem__(self, idx, value: ty.Optional[T]):  # idx could be a slice
        # if idx exists, replace the value; if not, raise an exception
        ...

    def __iter__(self):
        ...

    def __contains__(self, value):
        ...

    def __len__(self):
        return len(self._array)

    @property
    def shape(self) -> ty.Tuple[int, ...]:
        return self._index.shape

    def map(self, func: ty.Callable[T, ty.Any]) -> 'Field':
        """ apply `func` to every item """
        ...

    def filter(self, pred: ty.Callable[T, bool]) -> 'Field':
        """ unindex each element for which `pred` is False (in new Series) """
        ...

    def accum(self, accum_func) -> 'Field':
        ...


@attr.s(auto_attribs=True)
class DataFrame:
    """
    Rows are hierarchically indexed to provide arbitrary dimensionality.
    Each 'row' is has a fixed length and heterogeneous type.
    The collection of rows is of variable length and homogeneous type.
    Each 'column' is a nullable field, with every element the same type.
    Types:
    - boolean
    - categorical (enum)
    - integer
    - float
    - string
    - datetime
    - object (try not to use this)

    Encourage subclassing to add chained methods.
    """
    _fields: ty.Dict[str, Field]
    index: pd.MultiIndex

    @property
    def f(self):
        return Fields(self)

    @property
    def i(self):
        return self.index

    @property
    def r(self):
        return self.rows

    @property
    def rows(self) -> ty.Sequence[ty.Mapping]:
        return Rows(self)

    @classmethod
    def from_arrays(cls, arrays: ty.Mapping[str, ty.Sequence],
                    index: pd.MultiIndex, dtypes) -> 'DataFrame':
        ...

    @classmethod
    def from_records(cls, rows: ty.Iterable[ty.Mapping],
                     index: pd.MultIndex, dtypes) -> 'DataFrame':
        ...

    @classmethod
    def copy(cls, other: 'DataFrame') -> 'DataFrame':
        ...

    def __getitem__(self, idx_field):
        idx, field = idx_field[:-1], idx_field[-1]
        return self.f[field][idx]

    def __setitem__(self, idx_field, value):
        idx, field = idx_field[:-1], idx_field[-1]
        self.f[field][idx] = value

    @property
    def shape(self):
        return (*self.index.shape, len(self.fields))

    def join(self, other, how, left_on, right_on, suffixes=('_x', '_y'),
             validate=None) -> 'DataFrame':
        """ combine columns """
        ...

    def concat(self, other) -> 'DataFrame':
        """ combine rows """
        ...

    def assign(self, field_name, func) -> 'DataFrame':
        """ like r.map, but returns a new dataframe with one more column """
        new_df = DataFrame.copy(self)
        new_df._fields[field_name] = self.r.map(func)  # allow this coupling?
        return new_df


@attr.s(auto_attribs=True)  # should this be immutable?
class Fields:  # should this be nested in DataFrame?
    """ sliceable mapping of names to fields """
    _df: DataFrame

    def __getitem__(self, name) -> Field:  # name could be slice
        ...

    def __setitem__(self, name, field):  # field could be a dataframe
        ...

    def __delitem__(self, name):  # name could be a slice
        ...

    def __contains__(self, name):
        ...

    def __iter__(self):  # like dict.keys()?
        ...

    def __len__(self):
        ...


class Row:
    """ sliceable mapping of names to elements """
    def __getitem__(self, name):  # name could be a slice
        ...

    def __setitem__(self, name, value):  # value could be a mapping
        ...

    def __contains__(self, name):
        ...

    def __iter__(self):  # like dict.keys()?
        ...

    def __len__(self):
        ...


R = ty.Mapping  # R := row type


@attr.s(auto_attribs=True, slots=True, frozen=True)
class Rows(ty.Generic[R]):  # should this be nested in DataFrame?
    """ sequence of namedtuples """
    _df: DataFrame

    def map(self, func: ty.Callable[R, ty.Any]) -> Field:
        """ apply func to every row """
        ...

    def filter(self, pred: ty.Callable[R, bool]) -> DataFrame:
        """ unindex each row for which `pred` is False (in new df) """
        ...

    def sort(self, key):
        ...

    def __getitem__(self, idx) -> R:
        ...

    def __setitem__(self, idx, value: R):
        # allow new idx
        ...

    def __delitem__(self, idx):
        ...

    def __iter__(self):
        ...

    def __contains__(self, idx):
        ...

    def __len__(self):
        ...
