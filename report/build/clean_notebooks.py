"""Strip emojis from the notebooks and switch chart colours to a blue / black / white scheme."""
import re, sys
import nbformat

EMOJI = re.compile("[\U0001F300-\U0001FAFF☀-➿⌀-⏿⬀-⯿️‍]")

COLOR_MAP = [
    ('sns.set_theme(style="whitegrid", palette="viridis")', 'sns.set_theme(style="whitegrid", palette="Blues_d")'),
    ('color="#7fc97f"', 'color="#4a78b5"'), ('color="#fdc086"', 'color="#9db7d5"'),
    ('"seagreen"', '"#1f4e79"'), ('"darkorange"', '"#404040"'),
    ('palette="Greens_r"', 'palette="Blues_r"'), ('palette="Greens"', 'palette="Blues"'),
    ('palette="Oranges"', 'palette="Greys"'), ('palette="magma"', 'palette="Blues_r"'),
    ('palette="viridis"', 'palette="Blues_d"'), ("#b7e4c7", "#cfe0f3"),
    ('cmap="RdYlGn"', 'cmap=LinearSegmentedColormap.from_list("bwb", ["#111111", "#ffffff", "#1f4e79"])'),
    ('import seaborn as sns', 'import seaborn as sns\nfrom matplotlib.colors import LinearSegmentedColormap'),
    ('ax[1].set_title("Boxplot of Yield per Hectare (kg/ha)")',
     'ax[1].set_title("Boxplot of Yield per Hectare (kg/ha)"); ax[1].set_xlabel("kg / ha")'),
    ('plt.title("Key factors influencing paddy yield"); plt.xlabel("Importance (%)")',
     'plt.title("Key factors influencing paddy yield"); plt.xlabel("Importance (%)"); plt.ylabel("Feature")'),
    ('plt.title("Factors affecting yield PER HECTARE"); plt.xlabel("Importance (%)")',
     'plt.title("Factors affecting yield PER HECTARE"); plt.xlabel("Importance (%)"); plt.ylabel("Feature")'),
]

for path in sys.argv[1:]:
    nb = nbformat.read(path, 4)
    for c in nb.cells:
        src = EMOJI.sub("", c.source)
        src = re.sub(r"(?m)^(#+) +", r"\1 ", src)          # "##  Title" -> "## Title"
        src = re.sub(r"(?<=\S)  +(?=\S)", " ", src) if c.cell_type == "markdown" else src
        src = src.replace('print(f"', 'print(f"').replace('"  Recommended', '"Recommended')
        if c.cell_type == "code":
            for a, b in COLOR_MAP:
                src = src.replace(a, b)
            c.outputs, c.execution_count = [], None
        c.source = src
    nbformat.write(nb, path)
    left = [ch for c in nb.cells for ch in c.source if EMOJI.match(ch)]
    print(path, "emojis left:", len(left))
