import torch
from torch.utils import benchmark

# results = []
# for b in [10, 10000, 2000000]:
#     for n in [10, 100, 10000, 1000000]:
#         if b * n >= 1000000000:
#             continue
#
#         description = f'[{b}, {n}]'
#
#         x = torch.rand(b, n, device='cuda')
#         y = torch.rand(b, n, device='cuda')
#
#         results.append(benchmark.Timer(
#             stmt='(x * y).sum(dim=-1)',
#             globals={'x': x, 'y': y},
#             description=description,
#         ).blocked_autorange())
#
#         results.append(benchmark.Timer(
#             stmt='torch.einsum("...j,...j->...", x, y)',
#             globals={'x': x, 'y': y},
#             description=description,
#         ).blocked_autorange())
#
# compare = benchmark.Compare(results)
# compare.trim_significant_figures()
# compare.colorize()
# compare.print()
#
#
# results = []
# for b in [10, 100, 1000]:
#     for n in [10, 100, 10000, 1000000]:
#         if b * b * n >= 1000000000:
#             continue
#
#         description = f'[{b}, {b}, {n}]'
#
#         x = torch.rand(b, b, n, device='cuda')
#         y = torch.rand(b, b, n, device='cuda')
#
#         results.append(benchmark.Timer(
#             stmt='(x * y).sum(dim=-1)',
#             globals={'x': x, 'y': y},
#             description=description,
#         ).blocked_autorange())
#
#         results.append(benchmark.Timer(
#             stmt='torch.einsum("...j,...j->...", x, y)',
#             globals={'x': x, 'y': y},
#             description=description,
#         ).blocked_autorange())
#
# compare = benchmark.Compare(results)
# compare.trim_significant_figures()
# compare.colorize()
# compare.print()
#
#
# results = []
# for b in [10, 100, 1000]:
#     for n in [10, 100, 10000, 1000000]:
#         if b * b * n >= 1000000000:
#             continue
#
#         description = f'[{b}, {b}, {n}]'
#
#         x = torch.rand(b, 1, n, device='cuda')
#         y = torch.rand(1, b, n, device='cuda')
#
#         results.append(benchmark.Timer(
#             stmt='(x * y).sum(dim=-1)',
#             globals={'x': x, 'y': y},
#             description=description,
#         ).blocked_autorange())
#
#         results.append(benchmark.Timer(
#             stmt='torch.einsum("...j,...j->...", x, y)',
#             globals={'x': x, 'y': y},
#             description=description,
#         ).blocked_autorange())
#
# compare = benchmark.Compare(results)
# compare.trim_significant_figures()
# compare.colorize()
# compare.print()


results = []
for b in [622, 1322, 2002, 20000, 500000]:
    # for n in [9, 16, 3]:
    # if b * b * n >= 1000000000:
    #     continue

    description = f'[{b}, 16, 9]'

    x = torch.rand(b, 9, device='cuda')
    y = torch.rand(b, 16, 9, device='cuda')
    #
    # results.append(benchmark.Timer(
    #     stmt='(x * y).sum(dim=-1)',
    #     globals={'x': x, 'y': y},
    #     description=description,
    # ).blocked_autorange())

    results.append(benchmark.Timer(
        stmt='torch.einsum("zi,zui->zui", x, y)',
        globals={'x': x, 'y': y},
        description=description,
    ).blocked_autorange())

compare = benchmark.Compare(results)
compare.trim_significant_figures()
compare.colorize()
compare.print()