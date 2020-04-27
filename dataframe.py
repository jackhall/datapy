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
import copy
import functools as ft
import typing as ty

import attr
import numpy as np
import pandas as pd


IDX = ty.TypeVar('IDX')


class Index(ty.Protocol, ty.Collection[IDX], ty.Hashable):
    """ Encapsulates a mapping from user-specified index values to indicies
    to a np.ndarray: the `find` method. This mapping can be right-composed with
    others to create new indexes. The more convoluted the composition, the
    more expensive it will become to access the underlying numpy arrays, until
    the user decides to reshape.
    """
    def find(self, *obj):
        """ Returns whatever is needed to index a numpy array. """
        pass

    def fits_around(self, inner: 'Index') -> bool:
        """ True if this index can be composed with `inner` safely. """
        pass


class ComposeableIndexMixin:
    def fits_around(self: Index, inner: Index) -> bool:
        return all(self.find(idx) in inner for idx in self)

    def compose(self, inner: Index, verify: bool = False) -> 'Index':
        if verify and not self.fits_around(inner):
            raise IndexError('the domain of inner does not match the codomain of outer')

        new_index = copy.copy(self)

        @ft.wraps(new_index.find)
        def composed(idx):
            return inner.find(self.find(idx))

        new_index.find = composed
        return new_index


T = ty.TypeVar('T')


@attr.s(auto_attribs=True)
class Field(ty.Generic[T]):
    """ sequence of T """
    _array: np.ndarray = attr.ib()
    _null_mask: np.ndarray = attr.ib()
    _index: Index

    def __getitem__(self, idx) -> ty.Optional[T]:
        np_idx = self._index.find(idx)
        if self._null_mask[np_idx]:
            return self._array[np_idx]
        else:
            return None

    def __setitem__(self, idx, value: ty.Optional[T]):
        """ if idx exists, replace the value; if not, raise an exception """
        np_idx = self._index.find(idx)
        if value is None:
            self._null_mask[np_idx] = False
        else:
            self._array[np_idx] = value

    def __iter__(self):
        for idx in self._index:
            yield self[idx]

    def __contains__(self, value):
        ...

    def __len__(self):
        return len(self._index)

    def map(self, func: ty.Callable[[T], ty.Any]) -> 'Field':
        """ apply `func` to every item """
        new_array = copy.copy(self._array)
        for np_idx, value in np.ndenumerate(self._array):
            if self._null_mask[np_idx]:
                new_array[np_idx] = func(value)
        return attr.evolve(self, array=new_array)

    def filter(self, pred: ty.Callable[[T], bool]) -> 'Field':
        """ unindex each element for which `pred` is False (in new Series) """
        ...

    # for later
    # def accum(self, accum_func) -> 'Field':
    #     ...


@attr.s(auto_attribs=True)
class DataFrame:
    """
    Rows are hierarchically indexed to provide arbitrary dimensionality.
    Each 'row' is has a fixed length and heterogeneous type.
    The collection of rows is of variable length and homogeneous type.
    Each 'column' is a nullable field, with every element the same type.
    Types (similar to numpy):
    - boolean
    - categorical (enum)
    - integer
    - float
    - string
    - datetime
    - object (try not to use this)

    Encourage subclassing to add chained methods.
    """
    _fields: ty.MutableMapping[str, Field]  # an OrderedDict
    _index: Index

    @property
    def index(self):
        return self._index

    @property
    def fields(self):
        return Fields(self)

    @property
    def rows(self) -> ty.Sequence[ty.Mapping]:
        return Rows(self)

    i = index
    f = fields
    r = rows

    @classmethod
    def from_arrays(cls, arrays: ty.Mapping[str, ty.Sequence],
                    index: pd.MultiIndex, dtypes) -> 'DataFrame':
        ...

    @classmethod
    def from_records(cls, rows: ty.Iterable[ty.Mapping],
                     index: pd.MultIndex, dtypes) -> 'DataFrame':
        ...

    def copy(self) -> 'DataFrame':
        return attr.evolve(self, fields={
            name: copy.copy(field) for name, field in self._fields.items()
        })

    def reshape(self):
        """ reshape each field according to the current index """
        return DataFrame(
            fields={name: field.reshape(self._index)
                    for name, field in self._fields.items()},
            index=self._index.flatten(),
        )

    # for later:
    # def join(self, other, how, left_on, right_on, suffixes=('_x', '_y'),
    #          validate=None) -> 'DataFrame':
    #     """ combine columns """
    #     ...
    #
    # def concat(self, other) -> 'DataFrame':
    #     """ combine rows """
    #     ...
    #
    # def assign(self, field_name, func) -> 'DataFrame':
    #     """ like r.map, but returns a new dataframe with one more column """
    #     new_df = DataFrame.copy(self)
    #     new_df._fields[field_name] = self.r.map(func)  # allow this coupling?
    #     return new_df


@attr.s(auto_attribs=True, slots=True, frozen=True)
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
    """ sliceable mapping of names to elements
    Elements can be updated but not added or deleted.
    """
    def __getitem__(self, name):  # name could be a slice
        ...

    def __setitem__(self, name, value):  # value could be a mapping
        ...

    def __contains__(self, name):
        # If a row is a mapping, contains must refer to the field names.
        # But this does not sound like what a user would expect; it's
        # more intuitive if a row is a container of values rather than keys.
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
