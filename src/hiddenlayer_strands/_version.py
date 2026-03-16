from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("hiddenlayer-strands")
except PackageNotFoundError:
    __version__ = "unknown"
