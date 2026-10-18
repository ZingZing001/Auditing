"""Shared figure style for the final report: one semantic encoding per role, used by every plotting script.

Each role has its own colour, marker (filled or open) and line pattern, so a series can be identified in
colour, greyscale or by colour-blind readers. A panel compares one pair of roles; other factors (dataset,
model, depth) become panel titles. Figures are drawn at 6.30 in wide, the report's 16 cm text width,
so font sizes here are the printed sizes.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

WIDTH = 6.30
LONG, DASHDOT, REF, CHANCE = (0, (6, 3)), (0, (6, 2, 1.5, 2)), (0, (3, 2)), (0, (1, 2))
# role: (label, colour, marker, filled, line pattern)
ROLES = {
    'gap':      ('Coverage gap',          '#D55E00', 'o', True,  '-'),
    'random':   ('Random deletion',       '#0072B2', 's', False, LONG),
    'every':    ('Every point attacked',  '#009E73', '^', True,  '-'),
    'correct':  ('Correct only',          '#CC79A7', 'D', False, LONG),
    'tree':     ('Tree, white-box',       '#E69F00', '>', True,  '-'),
    'svm':      ('SVM, HopSkipJump',      '#332288', 'v', False, LONG),
    'before':   ('Before split',          '#999933', 's', False, LONG),
    'trainonly': ('Training only',        '#661100', 'o', True,  '-'),
    'measured': ('Measured accuracy',     '#AA4499', 'h', True,  '-'),
    'honest':   ('Honest accuracy',       '#222222', 'x', False, LONG),
}
plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Times New Roman', 'Times', 'STIXGeneral'], 'mathtext.fontset': 'stix',
    'font.size': 9.5, 'axes.labelsize': 11, 'axes.titlesize': 10.5, 'xtick.labelsize': 9.5, 'ytick.labelsize': 9.5,
    'legend.fontsize': 9.5, 'text.color': '#222222', 'axes.labelcolor': '#222222', 'axes.edgecolor': '#555555',
    'axes.linewidth': 0.7, 'xtick.color': '#222222', 'ytick.color': '#222222', 'xtick.major.width': 0.7,
    'ytick.major.width': 0.7, 'pdf.fonttype': 42, 'savefig.facecolor': 'white', 'lines.dash_capstyle': 'butt',
})


def _mstyle(role):
    _, c, m, filled, _ = ROLES[role]
    return dict(marker=m, ms=6 if m == 'x' else 5.5, mec=c if (not filled or m == 'x') else '#333333',
                mfc=c if filled else 'white', mew=1.3 if m == 'x' else 0.9)


def line(ax, x, mean, half, role, band=True):
    """Mean with a 95% band (mean +/- half); NaN cells stay as gaps."""
    _, c, _, _, ls = ROLES[role]
    x, mean, half = np.asarray(x, float), np.asarray(mean, float), np.asarray(half, float)
    if band:
        ax.fill_between(x, mean - half, mean + half, color=c, alpha=0.12, lw=0, zorder=1)
    ax.plot(x, mean, color=c, ls=ls, lw=1.8, zorder=3, **_mstyle(role))


def dot(ax, x, y, half, role):
    """Horizontal dot-and-interval (solid whisker; identity from marker and colour)."""
    c = ROLES[role][1]
    ax.errorbar(x, y, xerr=half, fmt='none', ecolor=c, elinewidth=1.2, capsize=2.5, capthick=1.2, zorder=2)
    ax.plot([x], [y], ls='none', zorder=3, **_mstyle(role))


def handle(role, whisker=False):
    label, c, _, _, ls = ROLES[role]
    return Line2D([], [], color=c, ls='-' if whisker else ls, lw=1.2 if whisker else 1.8, label=label, **_mstyle(role))


def legend(fig, rows, whisker=False):
    """Legend above the panels (constrained layout). `rows` is a list of role lists, or of
    (title, roles) pairs for a labelled multi-row legend such as 'Spread:' / 'Accuracy:'."""
    if isinstance(rows[0], str):
        rows = [(None, rows)]
    titled = rows[0][0] is not None
    width = max(len(r) for _, r in rows) + titled
    grid = []
    for title, roles in rows:
        row = ([Line2D([], [], ls='none', label=title)] if titled else []) + [handle(r, whisker) for r in roles]
        grid.append(row + [Line2D([], [], ls='none', label='')] * (width - len(row)))
    handles = [grid[r][c] for c in range(width) for r in range(len(grid))]   # legend fills column-major
    fig.legend(handles=handles, loc='outside upper center', ncol=width, frameon=False, handlelength=3.6,
               handletextpad=0.6, columnspacing=1.5)


def rowlabel(ax, text):
    """Bold row heading to the left of a panel's y label."""
    ax.annotate(text, xy=(0, 0.5), xycoords='axes fraction', xytext=(-48, 0), textcoords='offset points',
                rotation=90, ha='right', va='center', fontsize=10.5, weight='bold')


def refline(ax, y=None, x=None, chance=False):
    """Grey reference line: short dash for no-defect / equal ratios, dots for chance."""
    kw = dict(color='#666666', lw=1.0, ls=CHANCE if chance else REF, zorder=2)
    ax.axhline(y, **kw) if y is not None else ax.axvline(x, **kw)


def tidy(ax, title=None, letter=None, ygrid=True):
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.grid(axis='y' if ygrid else 'x', color='#DEDEDE', lw=0.45); ax.set_axisbelow(True)
    if title or letter:
        ax.set_title(' '.join(t for t in (f'({letter})' if letter else '', title or '') if t), loc='left', pad=4)


def share_ylim(axes, step):
    """One y range for panels showing the same metric, widened to whole steps; never clips a band."""
    lo = min(a.get_ylim()[0] for a in axes); hi = max(a.get_ylim()[1] for a in axes)
    for a in axes:
        a.set_ylim(np.floor(lo / step) * step, np.ceil(hi / step) * step)


def draw(ax, x, vals, role):
    """Plot per-level arrays of seed values as mean and 95% band; empty levels stay as gaps."""
    m = [np.nanmean(v) if np.any(~np.isnan(v)) else np.nan for v in vals]
    c = [ci(v) if np.sum(~np.isnan(v)) > 1 else np.nan for v in vals]
    line(ax, x, m, c, role)


def ci(v):
    v = np.asarray(v, float)
    return 1.96 * np.nanstd(v, ddof=1) / np.sqrt(np.sum(~np.isnan(v)))


if __name__ == '__main__':   # self-check: every role has a distinct colour and a distinct (marker, fill, pattern)
    cols = [r[1] for r in ROLES.values()]; assert len(set(cols)) == len(cols)
    for a, b in [('gap', 'random'), ('every', 'correct'), ('tree', 'svm'), ('before', 'trainonly'), ('measured', 'honest')]:
        ra, rb = ROLES[a], ROLES[b]; assert ra[2:] != rb[2:] and ra[3] != rb[3], (a, b)
    print('ok')
