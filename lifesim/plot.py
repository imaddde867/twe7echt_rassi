import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def make_plots(snaps, deaths, path, title=""):
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    if len(snaps):
        yrs = snaps.groupby("day")
        ax[0, 0].plot(yrs.size().index / 365, yrs.size().values)
        ax[0, 1].plot(yrs["cash"].median().index / 365, yrs["cash"].median().values)
        last = snaps[snaps["day"] == snaps["day"].max()]
        ax[1, 1].scatter(last["education"], last["cash"], s=12)
    if len(deaths):
        ax[1, 0].hist(deaths["age"], bins=20)
    ax[0, 0].set(title="Alive agents", xlabel="year")
    ax[0, 1].set(title="Median cash (alive)", xlabel="year")
    ax[1, 0].set(title="Age at death", xlabel="age")
    ax[1, 1].set(title="Final snapshot: education vs cash",
                 xlabel="education", ylabel="cash")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
