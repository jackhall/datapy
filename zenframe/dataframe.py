import copy
from dataclasses import dataclass, replace
import typing as ty

from .index import ComposeableIndex
from .field import IndexedNullableField


T = ty.TypeVar('T')


@dataclass
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
    _fields: ty.MutableMapping[str, IndexedNullableField]  # an OrderedDict
    _index: ComposeableIndex

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
                    index: ComposeableIndex, dtypes) -> 'DataFrame':
        ...

    @classmethod
    def from_records(cls, rows: ty.Iterable[ty.Mapping],
                     index: ComposeableIndex, dtypes) -> 'DataFrame':
        ...

    def copy(self) -> 'DataFrame':
        return replace(self, _fields={
            name: copy.copy(field) for name, field in self._fields.items
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


@dataclass(frozen=True)
class Fields:  # should this be nested in DataFrame?
    """ sliceable mapping of names to fields """
    _df: DataFrame

    def __getitem__(self, name) -> IndexedNullableField:  # name could be slice
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


R = ty.TypeVar("R")  # R := row type (mapping?)



@dataclass(frozen=True)
class Rows(ty.Generic[R]):  # should this be nested in DataFrame?
    """ sequence of namedtuples """
    _df: DataFrame

    def map(self, func: ty.Callable[[R], ty.Any]) -> IndexedNullableField:
        """ apply func to every row """
        ...

    def filter(self, pred: ty.Callable[[R], bool]) -> DataFrame:
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
