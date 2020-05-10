# zenframe
a simpler Python dataframe

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
