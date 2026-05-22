# Visualization Style Rules

Use these defaults for academic publishing-facing figures unless a figure-specific
request says otherwise.

```python
LINEWIDTH = 5
MARKERSIZE = 14
FONTSIZE = 40
LEGEND_FONTSIZE = 30
plt.rcParams["font.family"] = "Times New Roman"
```

On Linux systems without Times New Roman, choose the first installed family from
this Times-compatible fallback list:

```python
plt.rcParams["font.family"] = [
    "Times New Roman",
    "TeX Gyre Termes",
    "Nimbus Roman",
    "Tinos",
    "Liberation Serif",
]
```

- Prefer PDF outputs.
- Set axis labels explicitly.
- Set tick label size to `FONTSIZE` minus a small offset.
- Do not add figure suptitles or subplot titles unless specifically requested.
