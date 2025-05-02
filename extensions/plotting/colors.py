# colors and colormaps
tab10 = [
    "#76B7B2",
    "#FF9DA7",
    "#4E79A7",
    "#F28E2B",
    "#59A14F",
    "#E15759",
    "#B07AA1",
    "#EDC948",
    "#BAB0AC",
    "#9C755F",
]
tab2 = ["#499894", "#FF9DA7"]
tab3 = ["#499894", "#F28E2B", "#FF9DA7"]
tab4 = ["#499894", "#FF9DA7", "#E15759", "#9d3c3e"]
tab4_2 = [
    "#FF9DA7",
    "#AB414C",
    "#2880DE",
    "#4E79A7",
]

try:
    import cmap as cm
    import numpy as np

    _wpk = cm.Colormap(
        {
            "red": lambda x: 1 - (x ** (10 / 4)),
            "blue": lambda x: (np.tanh((10 / 13 - x) * 8) + 1.0) / 2.0,
            "green": lambda x: (np.tanh((8 / 13 - x) * 5) + 1.0) / 2.0,
        },
        name="wpk",
    )
    wpk = _wpk.to_mpl()

    _koy = cm.Colormap(
        {
            "red": lambda x: (np.tanh((-6 / 13 + x) * 6) + 1.0) / 2.0,
            "blue": lambda x: (np.tanh((-11 / 13 + x) * 4) + 1.0) / 2.0,
            "green": lambda x: x ** (7 / 3),
        },
        name="koy",
    )
    koy = _koy.to_mpl()

    _kgy = cm.Colormap(
        {
            "red": lambda x: (np.tanh((-8 / 13 + x) * 13) + 1.0) / 2.1,
            "green": lambda x: (x + ((x * (x - 1) * (x - 0.5) * (x - 0.65)) * 15))
            * (2.0 / 2.1),
            "blue": lambda x: (x / 10) + ((x * (x - 1) * (x - 0.2) * (x - 0.9)) * 8),
        },
        name="kgy",
    )
    kgy = _kgy.to_mpl()

    _magma = cm.Colormap(
        {
            # "red": lambda x: (np.tanh((-8/13+x)*13)+1.0)/2.1,
            "red": lambda x: (1 - (1 - x) ** 4) / 3
            + (np.tanh((-7 / 13 + x) * 8) + 1.0) / 3,
            # "green": lambda x: (x+((x*(x-1)*(x-0.5)*(x-0.65))*15))*(2.0/2.1),
            # "green": lambda x: (np.log2(1+x)/4)+(((80**x)-1)/79)/4*3,
            "green": lambda x: ((np.log2(1 + x * 63) / 6) / 20)
            + (((320**x) - 1) / 319) / 20 * 19,
            # "blue": lambda x: (x/10)+((x*(x-1)*(x-0.2)*(x-0.9))*8),
            "blue": lambda x: (np.sin(x * np.pi * 2) / 3) + x * 2 / 3,
        },
        name="magma",
    )
    magma = _magma.to_mpl()
except ImportError:
    pass
