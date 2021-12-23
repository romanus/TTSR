import pstats
from pstats import SortKey
p = pstats.Stats('prof-report')
# p.sort_stats(SortKey.TIME).print_stats(.03)
p.sort_stats(SortKey.CUMULATIVE).print_stats(.03)