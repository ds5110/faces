from .local_cache import LocalCache
from .alt_cache import AltCache
from .meta_cache import MetaCache

cache = LocalCache() # default cache
alt = AltCache() # default alternate cache
meta_cache = MetaCache(cache, alt) # default meta cache